import { bracketMinimum } from "./bracket-minimum";
import { GoldenSectionMinimizeStatus, goldenSectionMinimize } from "./golden-section-minimize";


export interface MinimizeOptions {
  tolerance?: number;
  initialIncrement?: number;
  lowerBound?: number;
  upperBound?: number;
  maxIterations?: number;
  guess?: number;
} 

export type MinimizeStatus = GoldenSectionMinimizeStatus;

export const minimizeGoldenSection1D = (f: (v: number) => number, options?: MinimizeOptions, status?: MinimizeStatus) => {
  options = options ?? {};
  let x0;
  const tolerance = options.tolerance === undefined ? 1e-8 : options.tolerance;
  const dx = options.initialIncrement === undefined ? 1 : options.initialIncrement;
  const xMin = options.lowerBound === undefined ? -Infinity : options.lowerBound;
  const xMax = options.upperBound === undefined ? Infinity : options.upperBound;
  const maxIterations = options.maxIterations === undefined ? 100 : options.maxIterations;
  const bounds: [number, number] = [0, 0];

  if (status) {
    status.iterations = 0;
    status.argmin = NaN;
    status.minimum = Infinity;
    status.converged = false;
  }

  if (isFinite(xMax) && isFinite(xMin)) {
    bounds[0] = xMin;
    bounds[1] = xMax;
  } else {
    // Construct the best guess we can:
    if (options.guess === undefined) {
      if (xMin > -Infinity) {
        x0 = xMax < Infinity ? 0.5 * (xMin + xMax) : xMin;
      } else {
        x0 = xMax < Infinity ? xMax : 0;
      }
    } else {
      x0 = options.guess;
    }

    bracketMinimum(bounds, f, x0, dx, xMin, xMax);

    if (isNaN(bounds[0]) || isNaN(bounds[1])) {
      return NaN;
    }
  }

  return goldenSectionMinimize(f, bounds[0], bounds[1], tolerance, maxIterations, status);
};
