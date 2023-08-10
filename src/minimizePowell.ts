import { minimizeGoldenSection1D } from "./minimizeGoldenSection1D";

interface PowellOptions {
    maxIter?: number;
    maxIterLinearSearch?: number;
    lineTolerance?: number;
    tolerance?: number;
    bounds?: ([number, number] | null)[];
    verbose?: boolean;
}

interface PowellStatus {
    points: number[][]
}

export const minimizePowell = async <T extends number[]>(f: (v: T) => (number | Promise<number>), x0: T, options?: PowellOptions, status?: PowellStatus): Promise<T | undefined> => {
  let i, j, iter, ui: number[], tmin, pj: number[], fi, un, u: number[][], p0, sum, err, perr, du, tlimit;

  const maxIter = options?.maxIter ?? 20
  const bounds = options?.bounds ?? []
  const verbose = options?.verbose ?? false
  const maxIterLinearSearch = options?.maxIterLinearSearch ?? 100;
  const dx = 0.1
  const tol = (options?.tolerance ?? 1e-8)
  const tol1d = (options?.lineTolerance ?? tol) * dx

  if (status) status.points = [];

  // Dimensionality:
  let n = x0.length;
  // Solution vector:
  let p = x0.slice(0);

  // Search directions:
  u = [];
  un = [];
  for (i = 0; i < n; i++) {
    u[i] = [];
    for (j = 0; j < n; j++) {
      u[i][j] = i === j ? 1 : 0;
    }
  }

  // Bound the input:
  function constrain (x: number[]) {
    for (let i = 0; i < bounds.length; i++) {
      let ibounds = bounds[i];
      if (!ibounds) continue;
      if (isFinite(ibounds[0])) {
        x[i] = Math.max(ibounds[0], x[i]);
      }
      if (isFinite(ibounds[1])) {
        x[i] = Math.min(ibounds[1], x[i]);
      }
    }
  }

  constrain(p);

  if (status) status.points.push(p.slice());

  let bound = options?.bounds
    ? function (p: number[], ui: number[]) {
      let upper = Infinity;
      let lower = -Infinity;

      for (let j = 0; j < n; j++) {
        let jbounds = bounds[j];
        if (!jbounds) continue;

        if (ui[j] !== 0) {
          if (jbounds[0] !== undefined && isFinite(jbounds[0])) {
            lower = (ui[j] > 0 ? Math.max : Math.min)(lower, (jbounds[0] - p[j]) / ui[j]);
          }

          if (jbounds[1] !== undefined && isFinite(jbounds[1])) {
            upper = (ui[j] > 0 ? Math.min : Math.max)(upper, (jbounds[1] - p[j]) / ui[j]);
          }
        }
      }

      return [lower, upper];
    }
    : function () {
      return [-Infinity, Infinity];
    };

  // A function to evaluate:
  pj = [];
  fi = function (t: number) {
    for (let i = 0; i < n; i++) {
      pj[i] = p[i] + ui[i] * t;
    }

    return f(pj as T);
  };

  iter = 0;
  perr = 0;
  while (++iter < maxIter) {

    // Reinitialize the search vectors:
    if (iter % (n) === 0) {
      for (i = 0; i < n; i++) {
        u[i] = [];
        for (j = 0; j < n; j++) {
          u[i][j] = i === j ? 1 : 0;
        }
      }
    }

    // Store the starting point p0:
    for (j = 0, p0 = []; j < n; j++) {
      p0[j] = p[j];
    }

    // Minimize over each search direction u[i]:
    for (i = 0; i < n; i++) {
      ui = u[i];
      // Compute bounds based on starting point p in the
      // direction ui:
      tlimit = bound(p, ui);

      tmin = await minimizeGoldenSection1D(fi, {
        lowerBound: tlimit[0],
        upperBound: tlimit[1],
        initialIncrement: dx,
        tolerance: tol1d,
        maxIterations: maxIterLinearSearch,
      });

      if (tmin === undefined) {
        return undefined
      }

      if (tmin === 0) {
        return p as T; 
      }

      // Update the solution vector:
      for (j = 0; j < n; j++) {
        p[j] += tmin * ui[j];
      }

      constrain(p);

      if (status) status.points.push(p.slice());
    }

    // Throw out the first search direction:
    u.shift();

    // Construct a new search direction:
    for (j = 0, un = [], sum = 0; j < n; j++) {
      un[j] = p[j] - p0[j];
      sum += un[j] * un[j];
    }
    // Normalize:
    sum = Math.sqrt(sum);

    if (sum > 0) {
      for (j = 0; j < n; j++) {
        un[j] /= sum;
      }
    } else {
      // Exactly nothing moved, so it it appears we've converged. In particular,
      // it's possible the solution is up against a boundary and simply can't
      // move farther.
      return p as T;
    }

    u.push(un);
    // One more minimization, this time along the new direction:
    ui = un;

    tlimit = bound(p, ui);

    tmin = await minimizeGoldenSection1D(fi, {
      lowerBound: tlimit[0],
      upperBound: tlimit[1],
      initialIncrement: dx,
      tolerance: tol1d,
      maxIterations: maxIterLinearSearch,
    });

    if (tmin === undefined) {
      return undefined;
    }

    if (tmin === 0 || !isFinite(tmin)) {
      return p as T;
    }

    err = 0;
    for (j = 0; j < n; j++) {
      du = tmin * ui[j];
      err += du * du;
      p[j] += du;
    }

    constrain(p);

    if (status) status.points.push(p.slice());

    err = Math.sqrt(err);

    if (verbose) console.log('Iteration ' + iter + ': ' + (err / perr) + ' f(' + p + ') = ' + await f(p as T));

    if (err / perr < tol) return p as T;

    perr = err;
  }

  return p as T;
}