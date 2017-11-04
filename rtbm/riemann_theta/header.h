/**
 * Collects all external functions used by cython
 * (eliminates compiler warnings)
 */


// riemann_theta.pyx
void finite_sum_without_derivatives(double*, double*, double*,
                                    double*, double*, double*,
                                    double*, double*, int, int, int);
void finite_sum_without_derivatives_phaseI(double*, double*, double*,
                                    double*, double*, double*,
                                    double*, double*, int, int, int);
void finite_sum_without_derivatives_phaseII(double*, double*, double*,
                                    double*, double*, double*,
                                    double*, double*, int, int, int);
void finite_sum_with_derivatives(double*, double*, double*, double*,
                                 double*, double*, double*, double*,
                                 double*, double*, int, int, int, int);
void finite_sum_with_derivatives_phaseI(double*, double*, double*, double*,
                                 double*, double*, double*, double*,
                                 double*, double*, int, int, int, int);
void finite_sum_with_derivatives_phaseII(double*, double*, double*, double*,
                                 double*, double*, double*, double*,
                                 double*, double*, int, int, int, int);

// radius.pyx
void lll_reduce(double*, int, double, double);
