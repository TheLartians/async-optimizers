
const PHI_RATIO = 2 / (1 + Math.sqrt(5));

export interface GoldenSectionMinimizeStatus {
  iterations?: number;
  argmin?: number
  minimum?: number
  converged?: boolean
}

export const goldenSectionMinimize = (f: (v: number) => number, xL: number, xU: number, tol: number, maxIterations: number, status?: GoldenSectionMinimizeStatus) => {
  let xF, fF;
  let iteration = 0;
  let x1 = xU - PHI_RATIO * (xU - xL);
  let x2 = xL + PHI_RATIO * (xU - xL);
  // Initial bounds:
  let f1 = f(x1);
  let f2 = f(x2);

  // Store these values so that we can return these if they're better.
  // This happens when the minimization falls *approaches* but never
  // actually reaches one of the bounds
  const f10 = f(xL);
  const f20 = f(xU);
  const xL0 = xL;
  const xU0 = xU;

  // Simple, robust golden section minimization:
  while (++iteration < maxIterations && Math.abs(xU - xL) > tol) {
    if (f2 > f1) {
      xU = x2;
      x2 = x1;
      f2 = f1;
      x1 = xU - PHI_RATIO * (xU - xL);
      f1 = f(x1);
    } else {
      xL = x1;
      x1 = x2;
      f1 = f2;
      x2 = xL + PHI_RATIO * (xU - xL);
      f2 = f(x2);
    }
  }

  xF = 0.5 * (xU + xL);
  fF = 0.5 * (f1 + f2);

  if (status) {
    status.iterations = iteration;
    status.argmin = xF;
    status.minimum = fF;
    status.converged = true;
  }

  if (isNaN(f2) || isNaN(f1) || iteration === maxIterations) {
    if (status) {
      status.converged = false;
    }
    return NaN;
  }

  if (f10 < fF) {
    return xL0;
  } else if (f20 < fF) {
    return xU0;
  } else {
    return xF;
  }
}
