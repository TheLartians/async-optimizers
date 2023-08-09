import { minimizePowell } from "./minimizePowell";

let i, j, k, n: number, sum, x0: number[];

const assertAlmostEqual = (computed: number | undefined, expected: number, tol?: number) => {
  expect(computed).toBeCloseTo(expected, -Math.log10((tol ?? 1e-6)))
}


function assertVectorAlmostEqual (computed: number[], expected: number[], tol?: number) {
  expect(computed.length).toEqual(expected.length)

  for (let i = 0; i < computed.length; i++) {
    assertAlmostEqual(computed[i], expected[i], tol);
  }
}

test('minimizes x^2 + y^2 - x * y starting at [-20, 25]', () => {
  assertVectorAlmostEqual(
    minimizePowell(x => {
      return 1 + x[0] * x[0] + x[1] * x[1] - 1.9 * x[0] * x[1];
    }, [-20, 25]),
    [0, 0]
  );
  
});

test('minimizes (x + 10)^2 + (y + 10)^2 to [-10, -10]', () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => { return Math.pow(x[0] - 10, 2) + Math.pow(x[1] - 10, 2); },
      [0, 0]
    ),
    [10, 10]
  );
  
});

test('minimizes (x + 10)^2 + (y + 10)^2 to [1, 1] within [0, 1] x [0, 1]', () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => { return Math.pow(x[0] - 10, 2) + Math.pow(x[1] - 10, 2); },
      [0.5, 0.5],
      {bounds: [[0, 1], [0, 1]]}
    ),
    [1, 1]
  );
  
});

test('minimizes (x - 10)^2 + (y - 10)^2 to [1, 1] within [0, 1] x [0, 1]', () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => { return Math.pow(x[0] + 10, 2) + Math.pow(x[1] + 10, 2); },
      [0.5, 0.5],
      {bounds: [[0, 1], [0, 1]]}
    ),
    [0, 0]
  );
  
});

test('minimizes (x - 10)^2 + (y - 10)^2 to [1, 1] for x within [0, 1]', () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => { return Math.pow(x[0] - 10, 2) + Math.pow(x[1] + 10, 2); },
      [0.5, 0.5],
      {bounds: [[0, 1], null]}
    ),
    [1, -10]
  );
  
});

test('Rosenbrock function', () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => {
        return 100.0 * Math.pow(x[1] - x[0] * x[0], 2) + Math.pow(x[0] - 1, 2);
      },
      [0.5, 0.5]
    ),
    [1, 1]
  );
  
});

test("Booth's function", () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => { return Math.pow(x[0] + 2 * x[1] - 7, 2) + Math.pow(2 * x[0] + x[1] - 5, 2); },
      [0.5, 0.5]
    ),
    [1, 3]
  );
  
});

test("Booth's function with one-way bounds", () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => { return Math.pow(x[0] + 2 * x[1] - 7, 2) + Math.pow(2 * x[0] + x[1] - 5, 2); },
      [0, 0],
      {bounds: [[-10, Infinity], [-Infinity, 10]]}
    ),
    [1, 3]
  );
  
});

test("Booth's function with two-way bounds", () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => { return Math.pow(x[0] + 2 * x[1] - 7, 2) + Math.pow(2 * x[0] + x[1] - 5, 2); },
      [0, 0],
      {bounds: [[-10, 10], [-10, 10]]}
    ),
    [1, 3]
  );
  
});

test("Booth's function with invalid initial guess", () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => { return Math.pow(x[0] + 2 * x[1] - 7, 2) + Math.pow(2 * x[0] + x[1] - 5, 2); },
      [100, 100],
      {bounds: [[-10, 10], [-10, 10]]}
    ),
    [1, 3]
  );
  
});

for (n = 0; n < 11; n++) {
  test('Rosenbrock function in ' + n + 'D', () => {
    assertVectorAlmostEqual(
      minimizePowell(
        x => {
          sum = 0;
          for (i = 0; i < x.length - 1; i++) {
            sum += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(x[i] - 1, 2);
          }
          return sum;
        },
        new Array(n).fill(0).map(function (d, i) { return i / 10; }),
        {maxIter: 10 + n * 8}
      ),
      new Array(n).fill(0).map(function () { return 1; }),
      1e-3
    );
    
  });
}

for (n = 0; n < 11; n++) {
  test('Rosenbrock function in ' + n + 'D in region [-10, 10]^n', () => {
    assertVectorAlmostEqual(
      minimizePowell(
        x => {
          sum = 0;
          for (i = 0; i < x.length - 1; i++) {
            sum += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(x[i] - 1, 2);
          }
          return sum;
        },
        new Array(n).fill(0).map(function (d, i) { return i / 10; }),
        {
          maxIter: 10 + n * 10,
          bounds: new Array(n).fill(0).map(function () { return [-10, 10]; })
        }
      ),
      new Array(n).fill(0).map(function () { return 1; }),
      1e-3
    );
    
  });
}

test("Beale's function", () => {
  const f = (x: [number,number]) => {
    return Math.pow(1.5 - x[0] + x[0] * x[1], 2) +
      Math.pow(2.25 - x[0] + x[0] * x[1] * x[1], 2) +
      Math.pow(2.625 - x[0] + x[0] * x[1] * x[1] * x[1], 2);
  };
  assertVectorAlmostEqual(
    minimizePowell(
      f,
      [2, 0],
      {bounds: [[-4.5, 4.5], [-4.5, 4.5]]}
    ),
    [3, 0.5]
  );
  
});

test('Matyas function', () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => {
        return 0.26 * (x[0] * x[0] + x[1] * x[1]) - 0.48 * x[0] * x[1];
      },
      [1, 1],
      {bounds: [[-10, 10], [-10, 10]]}
    ),
    [0, 0]
  );
  
});

test('Golstein-Price function', () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => {
        return (1 + Math.pow(x[0] + x[1] + 1, 2) * (19 - 14 * x[0] + 3 * x[0] * x[0] - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] * x[1])) *
          (30 + Math.pow(2 * x[0] - 3 * x[1], 2) * (18 - 32 * x[0] + 12 * x[0] * x[0] + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] * x[1]));
      },
      [0, 0],
      {bounds: [[-2.5, 2.5], [-2.5, 2.5]]}
    ),
    [0, -1]
  );
  
});

test('McCormick function', () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => {
        return Math.sin(x[0] + x[1]) + Math.pow(x[0] - x[1], 2) - 1.5 * x[0] + 2.5 * x[1] + 1;
      },
      [0, 0],
      {bounds: [[-1.5, 4], [-3, 4]]}
    ),
    [-0.54719, -1.54719],
    1e-4
  );
  
});

// Fails because the line search doesn't successfully locate the sharp edges with
// a good enough tolerance. It just assumes it's found a min and doesn't resolve
// it well enough to march in the right direction.
/* test('Bukin function No. 6', () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => {
        return 100 + Math.sqrt(Math.abs(x[1] - 0.01 * x[0] * x[0])) + 0.01 * Math.abs(x[0] + 10);
      },
      [-7, 0],
      {bounds: [[-15, -5], [-3, 3]]}
    ),
    [-10, 1]
  );
  
}); */

test('Three hump camel function', () => {
  assertVectorAlmostEqual(
    minimizePowell(
      x => {
        return 2 * x[0] * x[0] - 1.05 * Math.pow(x[0], 4) + Math.pow(x[0], 6) / 6 + x[0] * x[1] + x[1] * x[1];
      },
      [1, 1],
      {bounds: [[-5, 5], [-5, 5]]}
    ),
    [0, 0]
  );
  
});

for (i = -3; i <= 3; i++) {
  for (j = -3; j <= 3; j++) {
    for (k = -3; k <= 3; k++) {
      x0 = [i / 2, j / 2, k / 2];
      test('3D paraboloid starting from ' + x0 + ' in [-1, 1] x [-1, 1] x [-1, 1]', () => {
        assertVectorAlmostEqual(
          minimizePowell(
            x => { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; },
            x0,
            {
              bounds: [[-1, 1], [-1, 1], [-1, 1]],
              maxIter: 10,
              tolerance: 1e-14
            }
          ),
          [0, 0, 0],
          1e-15
        );
        
      });
    }
  }
}

for (i = -3; i <= 3; i++) {
  for (j = -3; j <= 3; j++) {
    for (k = -3; k <= 3; k++) {
      x0 = [i / 2, j / 2, k / 2];
      test('3D paraboloid starting from ' + x0 + ' without bounds', () => {
        assertVectorAlmostEqual(
          minimizePowell(
            x => { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; },
            x0,
            {maxIter: 10}
          ),
          [0, 0, 0],
          1e-8
        );
        
      });
    }
  }
}