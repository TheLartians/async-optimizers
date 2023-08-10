import { GoldenSectionMinimizeStatus } from "./golden-section-minimize";
import { minimizeGoldenSection1D } from "./minimizeGoldenSection1D";


const assertAlmostEqual = (computed: number | undefined, expected: number, tol?: number) => {
  expect(computed).toBeCloseTo(expected, -Math.log10((tol ?? 1e-6)))
}

describe('minimize',  () => {
  test('Minimizes 1 / (x - 1) in [0, 2]', () => {
    const status: GoldenSectionMinimizeStatus = {};
    assertAlmostEqual(minimizeGoldenSection1D((x) => 1 / (x - 1), {
      lowerBound: 0,
      upperBound: 2
    }, status), 1);
    expect(status.converged).toBeTruthy();
    
  });

  test('Minimizes -1 / (x - 1) in [0, 2]', () => {
    const status: GoldenSectionMinimizeStatus = {};
    assertAlmostEqual(minimizeGoldenSection1D((x) => -1 / (x - 1), {
      lowerBound: 0,
      upperBound: 2
    }, status), 1);
    expect(status.iterations).toEqual( 41);
    assertAlmostEqual(status.argmin, 1);
    expect(status.converged).toBeTruthy();
    
  });

  test('Bails out on unbounded minimization of -x^2', () => {
    const status: GoldenSectionMinimizeStatus = {};
    minimizeGoldenSection1D((x) => {
      return -x * x;
    }, undefined, status);
    expect(isNaN(status.argmin!)).toBeTruthy();
    expect(isNaN(status.minimum!)).toBeTruthy();
    expect(status.converged).toBeFalsy();
    
  });

  test('Succeeds out on bounded minimization of -x^2', () => {
    const status: GoldenSectionMinimizeStatus = {};
    const answer = minimizeGoldenSection1D((x) => {
      return -x * x;
    }, {lowerBound: -1, upperBound: 2}, status);
    assertAlmostEqual(answer, 2);
    assertAlmostEqual(status.argmin, 2);
    assertAlmostEqual(status.minimum, -4);
    expect(status.converged).toBeTruthy();
    
  });

  test('Minimizes sqrt(x) in [0, inf)', () => {
    assertAlmostEqual(minimizeGoldenSection1D(Math.sqrt, {
      lowerBound: 0
    }), 0);
    
  });

  test('Minimizes sqrt(|x|) in (-inf, inf)', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => Math.sqrt(Math.abs(x))), 0);
    
  });

  test('returns answer if tolerance not met', () => {
    const status: GoldenSectionMinimizeStatus = {};
    minimizeGoldenSection1D((x) => x * (x - 2), {tolerance: 0, maxIterations: 200}, status)
    expect(status.iterations).toEqual( 200);
    expect(status.minimum).toEqual( -1);
    assertAlmostEqual(status.argmin, 1);
    expect(status.converged).toBeFalsy();
  });

  test('minimizes a x(x-2) in (-inf, inf)', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2)), 1);
    
  });

  test('minimizes x(x-2) in [-6, inf)', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2), {
      lowerBound: -6
    }), 1);
    
  });

  test('minimizes x(x-2) in (-inf, -6]', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2), {
      upperBound: -6
    }), -6);
    
  });

  test('minimizes x(x-2) in [6, inf)', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2), {
      lowerBound: 6
    }), 6);
    
  });

  test('minimizes x(x-2) in [5, 6]', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2), {
      lowerBound: 5,
      upperBound: 6
    }), 5);
    
  });

  test('minimizes a cubic', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2) * (x - 1), {
      lowerBound: 0,
      upperBound: 3
    }), (3 + Math.sqrt(3)) / 3);
    
  });

  test('minimizes a cubic', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2) * (x - 1), {
      lowerBound: -3,
      upperBound: 3
    }), -3);
    
  });

  test('maximizes a cubic', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => -(x * (x - 2) * (x - 1)), {
      lowerBound: 0,
      upperBound: 3
    }), 3);
    
  });

  test('minimizes a cubic against bounds', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2) * (x - 1), {
      lowerBound: 5,
      upperBound: 6
    }), 5);
    
  });

  test('minimizes a cubic against one bound', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2) * (x - 1), {
      lowerBound: 5
    }), 5);
    
  });

  test('fails to minimize a cubic with no bounds', () => {
    expect(isNaN(minimizeGoldenSection1D((x) => x * (x - 2) * (x - 1))!)).toBeTruthy();
  });

  test('minimizes a parabola with small starting increment', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2), {
      guess: 0,
      initialIncrement: 1e-4
    }), 1);
    
  });

  test('minimizes a parabola with small starting increment and bounds', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x * (x - 2), {
      guess: 0,
      lowerBound: 0,
      upperBound: 1,
      initialIncrement: 1e-4
    }), 1);
    
  });

  test('minimizes cosine', () => {
    assertAlmostEqual(Math.cos(minimizeGoldenSection1D(Math.cos)!), -1);
    
  });

  test('minimizes cosine', () => {
    assertAlmostEqual(minimizeGoldenSection1D(Math.cos), Math.PI);
    
  });

  test('minimizes cosine', () => {
    assertAlmostEqual(minimizeGoldenSection1D(Math.cos), Math.PI);
    
  });

  test('minimizes cosine', () => {
    assertAlmostEqual(minimizeGoldenSection1D(Math.cos, {lowerBound: 0, upperBound: 1}), 1);
    
  });

  test('minimizes cosine', () => {
    assertAlmostEqual(minimizeGoldenSection1D(Math.cos, {guess: -3}), -Math.PI);
    
  });

  test('minimizes sine', () => {
    assertAlmostEqual(minimizeGoldenSection1D(Math.sin), -Math.PI * 0.5);
    
  });

  test('minimizes a line', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => x, {
      lowerBound: 0.5,
      upperBound: 1
    }), 0.5);
    
  });

  test('minimizes a cusp', () => {
    assertAlmostEqual(minimizeGoldenSection1D((x) => Math.sqrt(Math.abs(x - 5))), 5);
    
  });
});
