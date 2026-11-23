/*
 * D3Q19 lattice-Boltzmann kernel for sim.cht (sparse fluid-node storage, float32).
 *
 * Storage
 *   Only fluid nodes are stored (N nodes).  Populations are kept as a
 *   structure of arrays f[q*N + i] and in "deviation" form g_q = f_q - w_q, so
 *   that single precision resolves small velocities (round-off scales with
 *   the Mach number instead of with 1).
 *
 * Model
 *   - incompressible equilibrium (He & Luo 1997), rho0 = 1:
 *       g_q^eq = w_q [ rho' + 3 c.u + 4.5 (c.u)^2 - 1.5 u.u ],  u = sum c_q g_q
 *     pressure p = cs^2 rho', cs^2 = 1/3.
 *   - two-relaxation-time collision (Ginzburg): even (symmetric) part relaxed
 *     with omega+ (sets the viscosity), odd part with omega- chosen from the
 *     "magic" parameter Lambda = (1/omega+ - 1/2)(1/omega- - 1/2).  Lambda = 3/16
 *     puts the halfway bounce-back wall exactly half a cell from the node for
 *     any viscosity.
 *   - the trace of the non-equilibrium stress (bulk mode, moment e) is relaxed
 *     with its own rate omega_b, giving a bulk viscosity that damps the
 *     acoustic transients independent of the (tiny) shear viscosity.
 *   - eddy viscosity nut[i] (WALE, computed in lbm_wale(), or in-kernel
 *     Smagorinsky when csmag2 > 0) is added to the molecular viscosity.
 *
 * Streaming
 *   Fused pull: population q of node i at t+1 is read from src[(q-1)*N + i],
 *   an index into the *old* post-collision array.  For a fluid neighbour it
 *   points at (q, i - c_q); for a wall link at (opposite q, i), which is
 *   halfway bounce-back.  Open boundaries (inlet / outlet) are bounce-back
 *   pointers whose source value is edited by the Python driver between steps.
 */
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

typedef float real;
#define NQ 19
#define BS 32 /* nodes per SIMD block */

/* velocity set: opposite of q (q >= 1) is q+1 for odd q, q-1 for even q */
static const real CX[NQ] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
static const real CY[NQ] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
static const real CZ[NQ] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
#define W0 (1.0f / 3.0f)
#define WA (1.0f / 18.0f)
#define WD (1.0f / 36.0f)

int lbm_max_threads(void) { return omp_get_max_threads(); }

/*
 * One fused stream+collide step.
 *   fa   [19N] old post-collision populations (read)
 *   fb   [19N] new post-collision populations (written)
 *   src  [18N] pull pointers for q = 1..18
 *   rho  [N]   rho' = rho - 1 of the pre-collision state at t+1 (written)
 *   u    [3N]  velocity (SoA: ux[N], uy[N], uz[N]) (written)
 *   nut  [N]   eddy (+sponge) viscosity added to nu0 (read)
 *   par  [8]   nu0, lambda, om_min, omega_b, c2, limiter, collision, unused
 *     nu0       molecular viscosity (lattice units)
 *     omega_b   relaxation rate of the bulk (trace) mode; <= 0 -> use omega+
 *     c2        constant^2 of the in-kernel strain-based eddy viscosity, 0 = off
 *     limiter   0: classic Smagorinsky, nu_eff = nu + nut + c2 |S|
 *               1: cell-Reynolds limiter, nu_eff = max(nu + nut, c2 |S|)
 *               (|S| = sqrt(2 S:S) from the non-equilibrium stress, dx = 1)
 *     collision 0: TRT with magic parameter lambda (omega- >= om_min)
 *               1: regularised (Latt & Chopard 2006): the populations are
 *                  rebuilt from rho', u and the non-equilibrium stress only,
 *                  i.e. every ghost moment is reset to equilibrium each step.
 *               2: recursive regularised (Malaspinas 2015): as 1 plus the
 *                  third-order Hermite terms of the equilibrium and of the
 *                  non-equilibrium (obtained recursively from u and Pi).
 */
void lbm_step(const real *restrict fa, real *restrict fb, const int32_t *restrict src,
              real *restrict rho, real *restrict u, const real *restrict nut, int64_t N,
              const real *par, int nthreads)
{
    const real nu0 = par[0], lambda = par[1], om_min = par[2], omega_b = par[3], c2 = par[4];
    const int limiter = par[5] > 0.5f, regularised = (int)(par[6] + 0.5f);
    real *restrict ux = u, *restrict uy = u + N, *restrict uz = u + 2 * N;
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int64_t i0 = 0; i0 < N; i0 += BS) {
        const int n = (int)((N - i0 < BS) ? (N - i0) : BS);
        real g[NQ][BS] __attribute__((aligned(64)));
        real r[BS], vx[BS], vy[BS], vz[BS], op[BS], om[BS], ecor[BS];
        real sxx[BS], syy[BS], szz[BS], sxy[BS], sxz[BS], syz[BS], str[BS];
        real ta1[BS], tb1[BS], ta2[BS], tb2[BS], ta3[BS], tb3[BS];

        for (int b = 0; b < n; b++) g[0][b] = fa[i0 + b];
        for (int q = 1; q < NQ; q++) {
            const int32_t *restrict s = src + (int64_t)(q - 1) * N + i0;
#pragma omp simd
            for (int b = 0; b < n; b++) g[q][b] = fa[s[b]];
        }
#pragma omp simd
        for (int b = 0; b < n; b++) {
            const real a1 = g[1][b], a2 = g[2][b], a3 = g[3][b], a4 = g[4][b], a5 = g[5][b],
                       a6 = g[6][b], a7 = g[7][b], a8 = g[8][b], a9 = g[9][b], a10 = g[10][b],
                       a11 = g[11][b], a12 = g[12][b], a13 = g[13][b], a14 = g[14][b],
                       a15 = g[15][b], a16 = g[16][b], a17 = g[17][b], a18 = g[18][b];
            const real sa = a1 + a2 + a3 + a4 + a5 + a6;
            const real sd = a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18;
            const real rr = g[0][b] + sa + sd;
            const real x = a1 - a2 + a7 - a8 + a9 - a10 + a11 - a12 + a13 - a14;
            const real y = a3 - a4 + a7 - a8 - a9 + a10 + a15 - a16 + a17 - a18;
            const real z = a5 - a6 + a11 - a12 - a13 + a14 + a15 - a16 - a17 + a18;
            r[b] = rr; vx[b] = x; vy[b] = y; vz[b] = z;
            /* non-equilibrium momentum flux  Pi = sum c c g - (rho'/3 I + u u) */
            const real pxx = a1 + a2 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 - rr * W0 - x * x;
            const real pyy = a3 + a4 + a7 + a8 + a9 + a10 + a15 + a16 + a17 + a18 - rr * W0 - y * y;
            const real pzz = a5 + a6 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 - rr * W0 - z * z;
            const real pxy = a7 + a8 - a9 - a10 - x * y;
            const real pxz = a11 + a12 - a13 - a14 - x * z;
            const real pyz = a15 + a16 - a17 - a18 - y * z;
            const real t3 = (pxx + pyy + pzz) * W0;
            const real dxx = pxx - t3, dyy = pyy - t3, dzz = pzz - t3; /* deviatoric part */
            real tau = 3.0f * (nu0 + nut[i0 + b]) + 0.5f;
            if (c2 > 0.0f) {
                const real pn = sqrtf(dxx * dxx + dyy * dyy + dzz * dzz +
                                      2.0f * (pxy * pxy + pxz * pxz + pyz * pyz));
                /* |S| = 3 sqrt2 |Pi| / (2 tau_eff); solve for tau_eff */
                const real ts = 0.5f * (tau + sqrtf(tau * tau + 25.455844f * c2 * pn));
                const real tl = 0.25f + sqrtf(0.0625f + 6.363961f * c2 * pn);
                tau = limiter ? ((tau > tl) ? tau : tl) : ts;
            }
            const real o_p = 1.0f / tau;
            const real o_m0 = 1.0f / (lambda / (tau - 0.5f) + 0.5f);
            op[b] = o_p;
            om[b] = (o_m0 > om_min) ? o_m0 : om_min;
            const real ob = (omega_b > 0.0f) ? omega_b : o_p;
            ecor[b] = (ob - o_p) * (3.0f * t3) * (1.0f / 126.0f);
            /* regularised collisions: post-collision non-equilibrium stress Pi* */
            const real kd = 1.0f - o_p, kt = (1.0f - ob) * t3;
            const real qxx = kd * dxx + kt, qyy = kd * dyy + kt, qzz = kd * dzz + kt;
            const real qxy = kd * pxy, qxz = kd * pxz, qyz = kd * pyz;
            sxx[b] = 4.5f * qxx; syy[b] = 4.5f * qyy; szz[b] = 4.5f * qzz;
            sxy[b] = 9.0f * qxy; sxz[b] = 9.0f * qxz; syz[b] = 9.0f * qyz;
            str[b] = -1.5f * (qxx + qyy + qzz);
            /* third-order Hermite coefficients a_abb = u_a u_b u_b (equilibrium)
               + u_a Pi*_bb + 2 u_b Pi*_ab (recursive non-equilibrium) */
            const real axxy = x * x * y + y * qxx + 2.0f * x * qxy;
            const real ayzz = y * z * z + y * qzz + 2.0f * z * qyz;
            const real axzz = x * z * z + x * qzz + 2.0f * z * qxz;
            const real axyy = x * y * y + x * qyy + 2.0f * y * qxy;
            const real ayyz = y * y * z + z * qyy + 2.0f * y * qyz;
            const real axxz = x * x * z + z * qxx + 2.0f * x * qxz;
            ta1[b] = 13.5f * (axxy + ayzz); tb1[b] = 4.5f * (axxy - ayzz);
            ta2[b] = 13.5f * (axzz + axyy); tb2[b] = 4.5f * (axzz - axyy);
            ta3[b] = 13.5f * (ayyz + axxz); tb3[b] = 4.5f * (ayyz - axxz);
        }
        for (int b = 0; b < n; b++) {
            rho[i0 + b] = r[b]; ux[i0 + b] = vx[b]; uy[i0 + b] = vy[b]; uz[i0 + b] = vz[b];
        }
        if (regularised) {
            /* g*_q = w_q [ rho' + 3 c.u + 4.5 (cc - I/3):(uu + Pi*) + third-order terms ]
               third order (D3Q19 basis, Jacob et al. 2018): the six cubic Hermite
               polynomials H_aab = (c_a^2 - 1/3) c_b are paired into orthogonal
               sums / differences. */
            const real h3 = (regularised > 1) ? 1.0f : 0.0f;
            for (int q = 0; q < NQ; q++) {
                const real w = (q == 0) ? W0 : ((q < 7) ? WA : WD);
                const real cxq = CX[q], cyq = CY[q], czq = CZ[q];
                const real cxx = cxq * cxq, cyy = cyq * cyq, czz = czq * czq;
                const real cxy = cxq * cyq, cxz = cxq * czq, cyz = cyq * czq;
                const real hxxy = (cxx - W0) * cyq, hyzz = (czz - W0) * cyq;
                const real hxzz = (czz - W0) * cxq, hxyy = (cyy - W0) * cxq;
                const real hyyz = (cyy - W0) * czq, hxxz = (cxx - W0) * czq;
                const real A1 = h3 * (hxxy + hyzz), B1 = h3 * (hxxy - hyzz);
                const real A2 = h3 * (hxzz + hxyy), B2 = h3 * (hxzz - hxyy);
                const real A3 = h3 * (hyyz + hxxz), B3 = h3 * (hyyz - hxxz);
                real *restrict o = fb + (int64_t)q * N + i0;
#pragma omp simd
                for (int b = 0; b < n; b++) {
                    const real usq = vx[b] * vx[b] + vy[b] * vy[b] + vz[b] * vz[b];
                    const real cu = cxq * vx[b] + cyq * vy[b] + czq * vz[b];
                    const real neq = cxx * sxx[b] + cyy * syy[b] + czz * szz[b] + cxy * sxy[b] +
                                     cxz * sxz[b] + cyz * syz[b] + str[b];
                    const real third = A1 * ta1[b] + B1 * tb1[b] + A2 * ta2[b] + B2 * tb2[b] +
                                       A3 * ta3[b] + B3 * tb3[b];
                    o[b] = w * (r[b] + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq + neq + third);
                }
            }
            continue;
        }
        /* TRT: rest population is even only; bulk-mode vector e_0 = -30 */
        {
            real *restrict o0 = fb + i0;
#pragma omp simd
            for (int b = 0; b < n; b++) {
                const real usq = vx[b] * vx[b] + vy[b] * vy[b] + vz[b] * vz[b];
                const real eq = W0 * (r[b] - 1.5f * usq);
                o0[b] = g[0][b] - op[b] * (g[0][b] - eq) + 30.0f * ecor[b];
            }
        }
        /* pairs (q, q+1), q odd */
        for (int q = 1; q < NQ; q += 2) {
            const real w = (q < 7) ? WA : WD;
            const real ei = (q < 7) ? -11.0f : 8.0f;
            const real cxq = CX[q], cyq = CY[q], czq = CZ[q];
            real *restrict oa = fb + (int64_t)q * N + i0;
            real *restrict ob_ = fb + (int64_t)(q + 1) * N + i0;
#pragma omp simd
            for (int b = 0; b < n; b++) {
                const real usq = vx[b] * vx[b] + vy[b] * vy[b] + vz[b] * vz[b];
                const real cu = cxq * vx[b] + cyq * vy[b] + czq * vz[b];
                const real ga = g[q][b], gb = g[q + 1][b];
                const real eqp = w * (r[b] + 4.5f * cu * cu - 1.5f * usq);
                const real eqm = w * 3.0f * cu;
                const real dp = op[b] * (0.5f * (ga + gb) - eqp) + ei * ecor[b];
                const real dm = om[b] * (0.5f * (ga - gb) - eqm);
                oa[b] = ga - dp - dm;
                ob_[b] = gb - dp + dm;
            }
        }
    }
}

/*
 * WALE eddy viscosity (Nicoud & Ducros 1999) from finite differences of u.
 *   nb    [6N] face-neighbour node index (-x,+x,-y,+y,-z,+z); the node itself
 *         where there is no fluid neighbour
 *   gcode [3N] per-axis stencil code selecting the difference formula:
 *         0 both neighbours fluid (central), 1 wall on - side, 2 wall on + side,
 *         3 walls on both sides, 4 open on - side, 5 open on + side,
 *         6 wall -, open +, 7 open -, wall +, 8 no usable neighbour.
 *         Walls sit half a cell away with u = 0 (second-order one-sided formula).
 *   base  [N]  viscosity always added (outlet sponge), may be NULL
 *   nut   [N]  output: cw2 * WALE operator + base
 */
void lbm_wale(const real *restrict u, const int32_t *restrict nb, const uint8_t *restrict gcode,
              const real *restrict base, real *restrict nut, int64_t N, real cw2, int nthreads)
{
    /* derivative = cR*u[R] + c0*u[i] + cL*u[L] */
    static const real LUT[9][3] = {
        {0.5f, 0.0f, -0.5f}, {1.0f / 3.0f, 1.0f, 0.0f}, {0.0f, -1.0f, -1.0f / 3.0f},
        {0.0f, 0.0f, 0.0f},  {1.0f, -1.0f, 0.0f},       {0.0f, 1.0f, -1.0f},
        {0.0f, 2.0f, 0.0f},  {0.0f, -2.0f, 0.0f},       {0.0f, 0.0f, 0.0f}};
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int64_t i = 0; i < N; i++) {
        real G[3][3]; /* G[a][c] = d u_c / d x_a */
        for (int a = 0; a < 3; a++) {
            const int64_t L = nb[(int64_t)(2 * a) * N + i], R = nb[(int64_t)(2 * a + 1) * N + i];
            const real *c = LUT[gcode[(int64_t)a * N + i]];
            for (int k = 0; k < 3; k++) {
                const real *uc = u + (int64_t)k * N;
                G[a][k] = c[0] * uc[R] + c[1] * uc[i] + c[2] * uc[L];
            }
        }
        /* g_ij = du_i/dx_j = G[j][i];  g2 = g.g */
        real g2[3][3];
        for (int p = 0; p < 3; p++)
            for (int q = 0; q < 3; q++)
                g2[p][q] = G[0][p] * G[q][0] + G[1][p] * G[q][1] + G[2][p] * G[q][2];
        const real tr = (g2[0][0] + g2[1][1] + g2[2][2]) * (1.0f / 3.0f);
        real ss = 0.0f, sdsd = 0.0f;
        for (int p = 0; p < 3; p++)
            for (int q = 0; q < 3; q++) {
                const real s = 0.5f * (G[q][p] + G[p][q]);
                real sdv = 0.5f * (g2[p][q] + g2[q][p]);
                if (p == q) sdv -= tr;
                ss += s * s;
                sdsd += sdv * sdv;
            }
        const real num = sdsd * sqrtf(sdsd);                      /* (Sd:Sd)^(3/2) */
        const real den = ss * ss * sqrtf(ss) + sdsd * sqrtf(sqrtf(sdsd)); /* (S:S)^(5/2)+(Sd:Sd)^(5/4) */
        const real v = (den > 1e-30f) ? cw2 * num / den : 0.0f;
        nut[i] = v + (base ? base[i] : 0.0f);
    }
}

/*
 * Accumulate running sums for time averaging (double precision):
 *   acc[0]=rho', [1..3]=u, [4..9]=uxux,uyuy,uzuz,uxuy,uxuz,uyuz, [10]=nut
 */
void lbm_accumulate(const real *restrict rho, const real *restrict u, const real *restrict nut,
                    double *restrict acc, int64_t N, int nthreads)
{
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int64_t i = 0; i < N; i++) {
        const double x = u[i], y = u[N + i], z = u[2 * N + i];
        acc[i] += rho[i];
        acc[N + i] += x;
        acc[2 * N + i] += y;
        acc[3 * N + i] += z;
        acc[4 * N + i] += x * x;
        acc[5 * N + i] += y * y;
        acc[6 * N + i] += z * z;
        acc[7 * N + i] += x * y;
        acc[8 * N + i] += x * z;
        acc[9 * N + i] += y * z;
        acc[10 * N + i] += nut[i];
    }
}

/* Global monitors: out[0] = sum u^2/2 (kinetic energy), out[1] = max |u|, out[2] = sum rho'. */
void lbm_monitor(const real *restrict rho, const real *restrict u, int64_t N, double *out,
                 int nthreads)
{
    double ke = 0.0, um = 0.0, m = 0.0;
#pragma omp parallel for schedule(static) num_threads(nthreads) reduction(+ : ke, m) reduction(max : um)
    for (int64_t i = 0; i < N; i++) {
        const double q = (double)u[i] * u[i] + (double)u[N + i] * u[N + i] +
                         (double)u[2 * N + i] * u[2 * N + i];
        ke += 0.5 * q;
        m += rho[i];
        if (q > um) um = q;
    }
    out[0] = ke; out[1] = sqrt(um); out[2] = m;
}
