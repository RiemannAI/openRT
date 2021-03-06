/*=============================================================================

  finite_sum.c

  Efficiently computing the finite sum part of the Riemann theta function.

  Functions
  ---------
  exppart
  exppart_phaseII
  normpart
  normpart_phaseI
  normpart_phaseII
  finite_sum_without_derivatives
  deriv_prod
  deriv_prod_phaseI
  deriv_prod_phaseII
  finite_sum_with_derivatives
  finite_sum_with_derivatives_phaseI
  finite_sum_with_derivatives_phaseII
  finite_sum_without_derivatives_phaseI
  finite_sum_without_derivatives_phaseII
  finite_sum_with_derivatives_normalized_phaseI
  finite_sum_with_derivatives_normalized_phaseII
  finite_sum_with_multi_derivatives_normalized_phaseI
  finite_sum_with_multi_derivatives_normalized_phaseII
  finite_sum_with_multi_derivatives_phaseI
  finite_sum_with_multi_derivatives_phaseII

  Original Authors
  -------

  * Chris Swierczewski (@cswiercz) - September 2012, July 2016
  * Grady Williams (@gradyrw) - October 2012
  * Jeremy Upsal (@jupsal) - July 2016

  Mods and Phase I and II functions
  -------
  * Stefano Carrazza, Daniel Krefl - Nov 2017

  =============================================================================*/

#ifndef __FINITE_SUM_C__
#define __FINITE_SUM_C__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


  /******************************************************************************
  exppart
  -------

  A helper function for the finite sum functions. Computes

             2pi < (n-intshift), (1/2)X(n-intshift) + x >

  ******************************************************************************/
  double
  exppart(double* n, double* X, double* x, double* intshift, int g)
  {
    double tmp1[g];
    double tmp2[g];
    int i,j;

    // tmp1 = n - intshift
    for (i = 0; i < g; i++)
      tmp1[i] = n[i] - intshift[i];

    // tmp2 = (1/2)X*(n-intshift)
    double sum;
    for (i = 0; i < g; i++) {
      sum = 0;
      for (j = 0; j < g; j++)
        sum += X[i*g + j] * tmp1[j];

      tmp2[i] = sum/2.0;
    }

    // tmp2 = (1/2)*X(n-intshift) + x
    for (i = 0; i < g; i++)
      tmp2[i] = tmp2[i] + x[i];

    // ept = <tmp1,tmp2>
    double dot = 0;
    for (i = 0; i < g; i++)
      dot += tmp1[i]*tmp2[i];

    return 2* M_PI * dot;
  }


  /*
    Phase special (X = 0)
  */


  double
  exppart_phaseII(double* n, double* X, double* x, int g)
  {

    double dot = 0;
    int i;

    for (i = 0; i < g; i++)
      dot += n[i]*x[i];

    return 2* M_PI * dot;
  }



  /******************************************************************************
  normpart
  --------

  A helper function for the finite sum functions. Computes

             -pi * || T*(n+fracshift) ||^2

  ******************************************************************************/
  double
  normpart(double* n, double* T, double* fracshift, int g)
  {
    double tmp1[g];
    double tmp2[g];
    int i,j;

    // tmp1 = n + fracshift
    for (i = 0; i < g; i++)
      tmp1[i] = n[i] + fracshift[i];

    // tmp2 = T*(n+fracshift)
    double sum;
    for (i = 0; i < g; i++) {
      sum = 0;
      for (j = 0; j < g; j++)
	sum += T[i*g + j] * tmp1[j];

      tmp2[i] = sum;
    }

    // norm = || T*(n + fracshift) || ^ 2
    double norm = 0;
    for (i = 0; i < g; i++)
      norm += tmp2[i] * tmp2[i];

    return -M_PI * norm;
  }

  /*
    Phase I
  */
  double
  normpart_phaseI(double* n, double* T, double* fracshift, int g)
  {
    double tmp1[g];
    int i,j;

    // tmp1 = n + fracshift
    for (i = 0; i < g; i++) {
      tmp1[i] = n[i] + fracshift[i];
    }

    // tmp2 = T*(n+fracshift)
    double norm = 0;

    double sum;
    for (i = 0; i < g; i++) {
      sum = 0;
      for (j = 0; j < g; j++)
	sum += T[i*g + j] * tmp1[j];

      norm += sum*sum;
    }

    return -M_PI * norm;
  }


  /*
    Phase II
  */

  double
  normpart_phaseII(double* n, double* T, int g)
  {

    int i,j;

    // tmp2 = T*(n+fracshift)
    double norm = 0;
    double sum;
    for (i = 0; i < g; i++) {
      sum = 0;
      for (j = 0; j < g; j++) {
	sum += T[i*g + j] * n[j];
      }
      norm += sum*sum;
    }

    return -M_PI * norm;
  }




  /******************************************************************************
  finite_sum_without_derivatives
  ------------------------------

  Computes the real and imaginary parts of the finite sum.

  Parameters
  ----------
  X, Yinv, T : double[:]
      Row-major matrices such that the Riemann matrix, Omega is equal to (X +
      iY). T is the Cholesky decomposition of Y.
  x, y : double[:]
      The real and imaginary parts of the input vector, z.
  S : double[:]
      The set of points in ZZ^g over which to compute the finite sum
  g : int
      The dimension of the above matrices and vectors
  N : int
     The number of points in ZZ^g over which to compute the sum
     (= total number of elements in S / g)

  Returns
  -------
  * fsum_real, fsum_imag : double*
        The real and imaginary parts of the finite sum.

  ******************************************************************************/

  void
  finite_sum_without_derivatives(double* fsum_real, double* fsum_imag,
				 double* zr, double* zi, double* X,
				 double* Yinv, double* T, double* S,
				 int g, int N, int num_vectors)
  {
    // compute the finite sum for each z-vector
    //#pragma omp parallel for
    for (int kk = 0; kk < num_vectors; kk++)
      {
	double *x = &zr[kk*g];
	double *y = &zi[kk*g];

	// compute the shifted vectors: shift = Yinv*y as well as its integer and
	// fractional parts
	int k,j;
	double shift[g];
	double intshift[g];
	double fracshift[g];

	// compute the following:
	//   * shift = Yinv*y;
	//   * intshift = round(shift)  ( or should it be floor?!?)
	//   * fracshift = shift - intshift
	double sum;
	for (k = 0; k < g; k++) {
	  sum = 0;
	  for (j = 0; j < g; j++)
	    sum += Yinv[k*g + j] * y[j];

          shift[k] = sum;
	}

	for(k = 0; k < g; k++) {
	  intshift[k] = round(shift[k]);
	  fracshift[k] = shift[k] - intshift[k];
	}

	// compute the finite sum
	double real_total = 0, imag_total = 0;
	double ept, npt, cpt, spt;
	double* n;
	for(k = 0; k < N; k++) {
	  // the current point in S \subset ZZ^g
	  n = S + k*g;

	  // compute the "cosine" and "sine" parts of the summand
	  ept = exppart(n, X, x, intshift, g);
	  npt = exp(normpart(n, T, fracshift, g));
	  cpt = npt * cos(ept);
	  spt = npt * sin(ept);
	  real_total += cpt;
	  imag_total += spt;
	}

	//store values to poiners
	fsum_real[kk] = real_total;
	fsum_imag[kk] = imag_total;
      }
  }



  /*
   *   Phase I
   *   Simplified version for purely imaginary Q and z
   *   (returns purely real)
   */

  void
  finite_sum_without_derivatives_phaseI(double* fsum_real, double* fsum_imag,
					double* zr, double* zi, double* X,
					double* Yinv, double* T, double* S,
					int g, int N, int num_vectors)
  {
    // compute the finite sum for each z-vector
    for (int kk = 0; kk < num_vectors; kk++)
      {
	double *y = &zi[kk*g];

	// compute the shifted vectors: shift = Yinv*y as well as its integer and
	// fractional parts
	int k,j;
	double fracshift[g];

	// compute the following:
	//   * shift = Yinv*y;
	//   * intshift = round(shift)  ( or should it be floor?!?)
	//   * fracshift = shift - intshift
	double sum;
	for (k = 0; k < g; k++) {
	  sum = 0;
	  for (j = 0; j < g; j++) {
	    sum += Yinv[k*g + j] * y[j];
	  }
	  fracshift[k] = sum - round(sum);
	}

	// compute the finite sum
	double real_total = 0;
	double* n;
	for(k = 0; k < N; k++) {
	  // the current point in S \subset ZZ^g
	  n = S + k*g;

	  real_total += exp(normpart_phaseI(n, T, fracshift, g));
	}

	//store values to poiners
	fsum_real[kk] = real_total;
	fsum_imag[kk] = 0;
      }
  }





  /*
   *   Phase II
   *   Simplified version for purely imaginary Q and purely real z
   *
   */

  void
  finite_sum_without_derivatives_phaseII(double* fsum_real, double* fsum_imag,
					 double* zr, double* zi, double* X,
					 double* Yinv, double* T, double* S,
					 int g, int N, int num_vectors)
  {

    // Empty
    for (int kk = 0; kk < num_vectors; kk++) {
      fsum_real[kk] = 0;
      fsum_imag[kk] = 0;
    }

    double* n;
    int k;

    for(k = 0; k < N; k++) {
      // the current point in S \subset ZZ^g
      n = S + k*g;

      double npt = exp(normpart_phaseII(n, T, g));

      // Sum over dataset
      for (int kk = 0; kk < num_vectors; kk++)
	{
          double *x = &zr[kk*g];

          // compute the "cosine" and "sine" parts of the summand
          double ept = exppart_phaseII(n, X, x, g);

          //store values to poiners
          fsum_real[kk] += npt * cos(ept);
          fsum_imag[kk] += npt * sin(ept);
	}

    }
  }


  /******************************************************************************
  deriv_prod
  ----------

  Compute the real and imaginary parts of the product
             ___
             | |    2*pi*I <d, n-intshift>
                   | |
               d in derivs

  for a given n in ZZ^g.

  Parameters
  ----------
  n : double[:]
      An integer vector in the finite sum ellipsoid.
  intshift : double[:]
      The integer part of Yinv*y.
  deriv_real, deriv_imag : double[:]
      The real and imaginary parts of the derivative directional vectors.
  nderivs : int
      Number / order of derivatives.
  g : int
      Genus / dimension of the problem.

  Returns
  -------
  * dpr, dpi : double
        The real and imaginary parts of the "derivative product".
  ******************************************************************************/
  void
  deriv_prod(double* dpr, double* dpi,
	     double* n, double* intshift,
	     double* deriv_real, double* deriv_imag, int nderivs,
	     int g)
  {
    double nmintshift[g];
    double term_real, term_imag;
    double total_real, total_real_tmp;
    double total_imag, total_imag_tmp;
    int i,j;

    // compute n-intshift
    for (i = 0; i < g; i++)
      nmintshift[i] = n[i] - intshift[i];

    /*
      Computes the dot product of each directional derivative and nmintshift.
      Then it computes the product of the resulting complex scalars.
    */
    total_real = 1;
    total_imag = 0;
    for (i = 0; i < nderivs; i++) {
      term_real = 0;
      term_imag = 0;
      for (j = 0; j < g; j++) {
	term_real += deriv_real[j + g*i] * nmintshift[j];
	term_imag += deriv_imag[j + g*i] * nmintshift[j];
      }

      /*
	Multiplies the dot product that was just computed with the product of all
	the previous terms. Total_real is the resulting real part of the sum, and
	total_imag is the resulting imaginary part.
      */
      total_real_tmp = total_real * term_real - total_imag * term_imag;
      total_imag_tmp = total_real * term_imag + total_imag * term_real;
      total_real = total_real_tmp;
      total_imag = total_imag_tmp;
    }

    // Compute (2*pi*i)^(nderivs) * (total_real + total_imag*i)
    double pi_mult = pow(2*M_PI, (double)nderivs);

    /*
      Determines what the result of i^nderivs is, and performs the correct
      multiplication afterwards.
    */
    if (nderivs % 4 == 0) {
      dpr[0] = pi_mult * total_real;
      dpi[0] = pi_mult * total_imag;
    }
    else if (nderivs % 4 == 1) {
      dpr[0] = -pi_mult * total_imag;
      dpi[0] = pi_mult * total_real;
    }
    else if (nderivs % 4 == 2) {
      dpr[0] = -pi_mult * total_real;
      dpi[0] = -pi_mult * total_imag;
    }
    else if (nderivs % 4 == 3) {
      dpr[0] = pi_mult * total_imag;
      dpi[0] = -pi_mult * total_real;
    }
  }

  /*
    Phase I
  */
  void
  deriv_prod_phaseI(double* dpr, double* dpi,
		    double* n, double* intshift,
		    double* deriv_real, double* deriv_imag, int nderivs,
		    int g)
  {
    double nmintshift[g];
    double term_real;
    double total_real;

    int i,j;

    // compute n-intshift
    for (i = 0; i < g; i++) {
      nmintshift[i] = n[i] - intshift[i];
    }

    /*
      Computes the dot product of each directional derivative and nmintshift.
      Then it computes the product of the resulting complex scalars.
    */
    total_real = 1;
    for (i = 0; i < nderivs; i++) {
      term_real = 0;
      for (j = 0; j < g; j++) {
	term_real += deriv_real[j + g*i] * nmintshift[j];
      }

      /*
	Multiplies the dot product that was just computed with the product of all
	the previous terms. Total_real is the resulting real part of the sum, and
	total_imag is the resulting imaginary part.
      */
      total_real = total_real * term_real;
    }



    // Compute (2*pi*i)^(nderivs) * (total_real + total_imag*i)
    double pi_mult = pow(2*M_PI, (double)nderivs);

    /*
      Determines what the result of i^nderivs is, and performs the correct
      multiplication afterwards.
    */
    if (nderivs % 4 == 0) {
      dpr[0] = pi_mult * total_real;
    }
    else if (nderivs % 4 == 1) {
      dpi[0] = pi_mult * total_real;
    }
    else if (nderivs % 4 == 2) {
      dpr[0] = -pi_mult * total_real;
    }
    else if (nderivs % 4 == 3) {
      dpi[0] = -pi_mult * total_real;
    }
  }


  /*
    Phase II
  */

  void
  deriv_prod_phaseII(double* dpr, double* dpi,
		     double* n,
		     double* deriv_real, double* deriv_imag, int nderivs,
		     int g)
  {

    double term_real;
    double total_real;

    int i,j;


    /*
      Computes the dot product of each directional derivative and nmintshift.
      Then it computes the product of the resulting complex scalars.
    */
    total_real = 1;

    for (i = 0; i < nderivs; i++) {
      term_real = 0;
      for (j = 0; j < g; j++) {
	term_real += deriv_real[j + g*(nderivs-1)] * n[j];
      }

      /*
	Multiplies the dot product that was just computed with the product of all
	the previous terms. Total_real is the resulting real part of the sum, and
	total_imag is the resulting imaginary part.
      */
      total_real = total_real * term_real;// - total_imag * term_imag;
    }


    // Compute (2*pi*i)^(nderivs) * (total_real + total_imag*i)
    double pi_mult = pow(2*M_PI, (double)nderivs);

    /*
      Determines what the result of i^nderivs is, and performs the correct
      multiplication afterwards.
    */
    if (nderivs % 4 == 0) {
      dpr[0] = pi_mult * total_real;
    }
    else if (nderivs % 4 == 1) {
      dpi[0] = pi_mult * total_real;
    }
    else if (nderivs % 4 == 2) {
      dpr[0] = -pi_mult * total_real;
    }
    else if (nderivs % 4 == 3) {
      dpi[0] = -pi_mult * total_real;
    }
  }




  /******************************************************************************
  finite_sum_with_derivatives
  ------------------------------

  Computes the real and imaginary parts of the finite sum with derivatives.

  Parameters
  ----------
  X, Yinv, T : double[:]
      Row-major matrices such that the Riemann matrix, Omega is equal to (X +
      iY). T is the Cholesky decomposition of Y.
  x, y : double[:]
      The real and imaginary parts of the input vector, z.
  S : double[:]
      The set of points in ZZ^g over which to compute the finite sum
  deriv_real, deriv_imag : double[:]
      The real and imaginary parts of the derivative directional vectors.
  nderivs : int
      Number / order of derivatives.
  g : int
      The dimension of the above matrices and vectors
  N : int
     The number of points in ZZ^g over which to compute the sum
     (= total number of elements in S / g)

  Returns
  -------
  fsum_real, fsum_imag : double*
      The real and imaginary parts of the finite sum.

  ******************************************************************************/
  void
  finite_sum_with_derivatives(double* fsum_real, double* fsum_imag,
			      double* X, double* Yinv, double* T,
			      double* zr, double* zi, double* S,
			      double* deriv_real, double* deriv_imag,
			      int nderivs, int g, int N, int num_vectors)
  {
    /*
      compute the shifted vectors: shift = Yinv*y as well as its integer and
      fractional parts
    */
    //#pragma omp parallel for
    for (int kk = 0; kk < num_vectors; kk++)
      {
	double *x = &zr[kk*g];
	double *y = &zi[kk*g];

	int k,j;
	double shift[g];
	double intshift[g];
	double fracshift[g];
	double sum;
	for (k = 0; k < g; k++) {
	  sum = 0;
	  for (j = 0; j < g; j++)
	    sum += Yinv[k*g + j] * y[j];

          shift[k] = sum;
	}

	for(k = 0; k < g; k++) {
	  intshift[k] = round(shift[k]);
	  fracshift[k] = shift[k] - intshift[k];
	}

	// compute the finite sum
	double real_total = 0, imag_total = 0;
	double ept, npt, cpt, spt;
	double dpr[1];
	double dpi[1];
	double* n;
	dpr[0] = 0;
	dpi[0] = 0;
	for(k = 0; k < N; k++) {
	  // the current point in S \subset ZZ^g
	  n = S + k*g;

	  // compute the "cosine" and "sine" parts of the summand
	  ept = exppart(n, X, x, intshift, g);
	  npt = exp(normpart(n, T, fracshift, g));
	  cpt = npt * cos(ept);
	  spt = npt * sin(ept);
	  deriv_prod(dpr, dpi, n, intshift, deriv_real, deriv_imag, nderivs, g);
	  real_total += dpr[0] * cpt - dpi[0] * spt;
	  imag_total += dpr[0] * spt + dpi[0] * cpt;
	}

	// store values to poiners
	fsum_real[kk] = real_total;
	fsum_imag[kk] = imag_total;
      }
  }


  /*
   *   Phase I
   *   Simplified version for purely imaginary Q and z
   *
   */

  void
  finite_sum_with_derivatives_phaseI(double* fsum_real, double* fsum_imag,
				     double* X, double* Yinv, double* T,
				     double* zr, double* zi, double* S,
				     double* deriv_real, double* deriv_imag,
				     int nderivs, int g, int N, int num_vectors)
  {
    /*
      compute the shifted vectors: shift = Yinv*y as well as its integer and
      fractional parts
    */

    // ToDo: Calc first shifts then sum over points !

    // Loop over dataset
    for (int kk = 0; kk < num_vectors; kk++)
      {
	double *y = &zi[kk*g];

	int k,j;

	double intshift[g];
	double fracshift[g];
	double sum;

	for (k = 0; k < g; k++) {
          sum = 0;
          for (j = 0; j < g; j++)
	    sum += Yinv[k*g + j] * y[j];

          intshift[k] = round(sum);
          fracshift[k] = sum - intshift[k];
	}


	// compute the finite sum
	double real_total = 0, imag_total = 0;
	double npt;
	double dpr[1];
	double dpi[1];
	double* n;
	dpr[0] = 0;
	dpi[0] = 0;

	for(k = 0; k < N; k++) {
	  // the current point in S \subset ZZ^g
	  n = S + k*g;

	  // compute the "cosine" and "sine" parts of the summand

	  npt = exp(normpart_phaseI(n, T, fracshift, g));

	  deriv_prod_phaseI(dpr, dpi, n, intshift, deriv_real, deriv_imag, nderivs, g);

	  real_total += dpr[0] * npt;
	  imag_total += dpi[0] * npt;
	}

	// store values to poiners
	fsum_real[kk] = real_total;
	fsum_imag[kk] = imag_total;
      }
  }


  /*
   *   Phase II
   *   Simplified version for purely imaginary Q and purely real z
   *
   */

  void
  finite_sum_with_derivatives_phaseII(double* fsum_real, double* fsum_imag,
				      double* X, double* Yinv, double* T,
				      double* zr, double* zi, double* S,
				      double* deriv_real, double* deriv_imag,
				      int nderivs, int g, int N, int num_vectors)
  {

    // Empty
    for (int kk = 0; kk < num_vectors; kk++) {
      fsum_real[kk] = 0;
      fsum_imag[kk] = 0;
    }

    double* n;
    double dpr[1];
    double dpi[1];

    for(int k = 0; k < N; k++) {
      // the current point in S \subset ZZ^g
      n = S + k*g;
      double npt = exp(normpart_phaseII(n, T, g));

      dpr[0] = 0;
      dpi[0] = 0;

      deriv_prod_phaseII(dpr, dpi, n, deriv_real, deriv_imag, nderivs, g);

      // Loop over dataset
      for (int kk = 0; kk < num_vectors; kk++)
	{
          double *x = &zr[kk*g];

          // compute the finite sum
          double ept, cpt, spt;

          // compute the "cosine" and "sine" parts of the summand
          ept = exppart_phaseII(n, X, x, g);
          cpt = npt*cos(ept);
          spt = npt*sin(ept);

          fsum_real[kk] += (dpr[0] * cpt - dpi[0] * spt);
          fsum_imag[kk] += (dpr[0] * spt + dpi[0] * cpt);
	}

    }
  }


  /*
   *   Phase I
   *   nth derivative over 0th derivative
   *
   */
  void
  finite_sum_with_derivatives_normalized_phaseI(double* fsum_real, double* fsum_imag,
						double* X, double* Yinv, double* T,
						double* zr, double* zi, double* S,
						double* deriv_real, double* deriv_imag,
						int nderivs, int g, int N, int num_vectors)
  {
    /*
      compute the shifted vectors: shift = Yinv*y as well as its integer and
      fractional parts
    */


    // Loop over dataset
    for (int kk = 0; kk < num_vectors; kk++)
      {
	double *y = &zi[kk*g];

	int k,j;

	double intshift[g];
	double fracshift[g];
	double sum;

	for (k = 0; k < g; k++) {
          sum = 0;
          for (j = 0; j < g; j++)
	    sum += Yinv[k*g + j] * y[j];

          intshift[k] = round(sum);
          fracshift[k] = sum - intshift[k];
	}


	// compute the finite sum
	double real_total_nom = 0, imag_total_nom = 0;
	double real_total_den = 0;

	double npt;
	double dpr[1];
	double dpi[1];
	double* n;
	dpr[0] = 0;
	dpi[0] = 0;

	for(k = 0; k < N; k++) {
	  // the current point in S \subset ZZ^g
	  n = S + k*g;

	  // compute the "cosine" and "sine" parts of the summand

	  npt = exp(normpart_phaseI(n, T, fracshift, g));

	  deriv_prod_phaseI(dpr, dpi, n, intshift, deriv_real, deriv_imag, nderivs, g);

	  real_total_nom += dpr[0] * npt;
	  imag_total_nom += dpi[0] * npt;
	  real_total_den += npt;
	}

	fsum_real[kk] = real_total_nom/real_total_den;
	fsum_imag[kk] = imag_total_nom/real_total_den;
      }
  }


  /*
   *   Phase II
   *   Simplified version for purely imaginary Q and purely real z
   *
   */

  void
  finite_sum_with_derivatives_normalized_phaseII(double* fsum_real, double* fsum_imag,
						 double* X, double* Yinv, double* T,
						 double* zr, double* zi, double* S,
						 double* deriv_real, double* deriv_imag,
						 int nderivs, int g, int N, int num_vectors)
  {

    // Allocate temp storage
    double norm_real[num_vectors];
    double norm_imag[num_vectors];

    // Empty
    for (int kk = 0; kk < num_vectors; kk++) {
      fsum_real[kk] = 0;
      fsum_imag[kk] = 0;
      norm_real[kk] = 0;
      norm_imag[kk] = 0;
    }

    double* n;
    double dpr[1];
    double dpi[1];

    for(int k = 0; k < N; k++) {
      // the current point in S \subset ZZ^g
      n = S + k*g;
      double npt = exp(normpart_phaseII(n, T, g));

      dpr[0] = 0;
      dpi[0] = 0;

      deriv_prod_phaseII(dpr, dpi, n, deriv_real, deriv_imag, nderivs, g);

      // Loop over dataset
      for (int kk = 0; kk < num_vectors; kk++)
	{
          double *x = &zr[kk*g];

          // compute the finite sum
          double ept, cpt, spt;

          // compute the "cosine" and "sine" parts of the summand
          ept = exppart_phaseII(n, X, x, g);
          cpt = npt*cos(ept);
          spt = npt*sin(ept);

          fsum_real[kk] += (dpr[0] * cpt - dpi[0] * spt);
          fsum_imag[kk] += (dpr[0] * spt + dpi[0] * cpt);
          norm_real[kk] += cpt;
          norm_imag[kk] += spt;
	}
    }

    // Loop over dataset (setting normalization)
    for (int kk = 0; kk < num_vectors; kk++)
      {
	double old_fsum_real = fsum_real[kk];
	double norm = norm_imag[kk]*norm_imag[kk]+norm_real[kk]*norm_real[kk];

	fsum_real[kk] = (fsum_real[kk]*norm_real[kk]+fsum_imag[kk]*norm_imag[kk])/norm;
	fsum_imag[kk] = (fsum_imag[kk]*norm_real[kk]-old_fsum_real*norm_imag[kk])/norm;
      }

  }


  /*
   *   Phase I
   *   multi nth derivative over 0th derivative
   *
   */
  void
  finite_sum_with_multi_derivatives_normalized_phaseI(double* fsum_real, double* fsum_imag,
						      double* Yinv, double* T, double* zi, double* S,
						      double* deriv_real_in, double* deriv_imag_in, int* n_derivs,
						      int numderivs, int g, int N, int num_vectors)
  {
    // Init
    int offset = 0;
    int nderivs[numderivs];
    double *deriv_real[numderivs];
    double *deriv_imag[numderivs];

    for(int d = 0; d < numderivs; d++) {
      nderivs[d] = n_derivs[d] / g;
      deriv_real[d] = &deriv_real_in[offset];
      deriv_imag[d] = &deriv_imag_in[offset];
      offset += n_derivs[d];

      // Empty
      for (int kk = 0; kk < num_vectors; kk++) {
        fsum_real[kk + d*num_vectors] = 0;
        fsum_imag[kk + d*num_vectors] = 0;
      }
    }

    // Loop over dataset
    for (int kk = 0; kk < num_vectors; kk++)
      {
        double *y = &zi[kk*g];

        int k,j;

        double intshift[g];
        double fracshift[g];
        double sum;

        for (k = 0; k < g; k++) {
	  sum = 0;
	  for (j = 0; j < g; j++) {
	    sum += Yinv[k*g + j] * y[j];
	  }

	  intshift[k] = round(sum);
	  fracshift[k] = sum - intshift[k];
        }

        // Init
        double real_total_den = 0;
        double real_total_nom[numderivs];
        double imag_total_nom[numderivs];

        for(int d = 0; d < numderivs; d++) {
	  real_total_nom[d] = 0;
	  imag_total_nom[d] = 0;
        }

        double npt;
        double dpr[1];
        double dpi[1];
        double* n;

        for(k = 0; k < N; k++) {

	  n = S + k*g;

	  npt = exp(normpart_phaseI(n, T, fracshift, g));
	  real_total_den += npt;

	  for (int d = 0; d < numderivs; d++)
	    {
	      if (n_derivs[d] > 0)
		{
		  dpr[0] = 0;
		  dpi[0] = 0;

		  deriv_prod_phaseI(dpr, dpi, n, intshift, deriv_real[d], deriv_imag[d], nderivs[d], g);

		  real_total_nom[d] += dpr[0] * npt;
		  imag_total_nom[d] += dpi[0] * npt;

		} else {
		real_total_nom[d] += npt;
		imag_total_nom[d] += npt;
	      }
	    }
        }

        // Store for return
        for (int d = 0; d < numderivs; d++)
	  {
            fsum_real[kk + d*num_vectors] = real_total_nom[d]/real_total_den;
            fsum_imag[kk + d*num_vectors] = imag_total_nom[d]/real_total_den;
	  }
      }
  }


  /*
   *   Phase II
   *   Simplified version for purely imaginary Q and purely real z
   *
   */

  void
  finite_sum_with_multi_derivatives_normalized_phaseII(double* fsum_real, double* fsum_imag,
						       double* X, double* T, double* zr, double* S,
						       double* deriv_real_in, double* deriv_imag_in, int* n_derivs,
						       int numderivs, int g, int N, int num_vectors)
  {

    int offset = 0;
    int nderivs[numderivs];
    double *deriv_real[numderivs];
    double *deriv_imag[numderivs];
    double dpr[numderivs][1];
    double dpi[numderivs][1];
    double norm_real[num_vectors];
    double norm_imag[num_vectors];

    for(int d = 0; d < numderivs; d++) {
      nderivs[d] = n_derivs[d] / g;
      deriv_real[d] = &deriv_real_in[offset];
      deriv_imag[d] = &deriv_imag_in[offset];
      offset += n_derivs[d];

      // Empty
      for (int kk = 0; kk < num_vectors; kk++) {
        fsum_real[kk + d*num_vectors] = 0;
        fsum_imag[kk + d*num_vectors] = 0;
      }
    }

    for (int kk = 0; kk < num_vectors; kk++)
    {
        norm_real[kk] = 0;
        norm_imag[kk] = 0;
    }

    double* n;
    double npt, ept, cpt, spt;

    for(int k = 0; k < N; k++) {
      // the current point in S \subset ZZ^g
      n = S + k*g;
      npt = exp(normpart_phaseII(n, T, g));

      for (int d = 0; d < numderivs; d++) {
        dpr[d][0] = 0;
        dpi[d][0] = 0;
        deriv_prod_phaseII(dpr[d], dpi[d], n, deriv_real[d], deriv_imag[d], nderivs[d], g);
      }

      // Loop over dataset
      for (int kk = 0; kk < num_vectors; kk++)
      {
          double *x = &zr[kk*g];

          // compute the "cosine" and "sine" parts of the summand
          ept = exppart_phaseII(n, X, x, g);
          cpt = npt*cos(ept);
          spt = npt*sin(ept);

          norm_real[kk] += cpt;
          norm_imag[kk] += spt;

          for (int d = 0; d < numderivs; d++)
          {
              if (n_derivs[d] > 0)
              {
                  fsum_real[kk + d*num_vectors] += (dpr[d][0] * cpt - dpi[d][0] * spt);
                  fsum_imag[kk + d*num_vectors] += (dpr[d][0] * spt + dpi[d][0] * cpt);
              }
              else
              {
                  fsum_real[kk + d*num_vectors] += cpt;
                  fsum_imag[kk + d*num_vectors] += spt;

              }
          }
       }
    }

    // Loop over dataset (setting normalization)
    for (int kk = 0; kk < num_vectors; kk++)
    {
        double norm = norm_imag[kk]*norm_imag[kk]+norm_real[kk]*norm_real[kk];

        for (int d = 0; d < numderivs; d++)
        {
            double old_fsum_real = fsum_real[kk + d*num_vectors];

            fsum_real[kk + d*num_vectors] = (fsum_real[kk + d*num_vectors]*norm_real[kk]+fsum_imag[kk + d*num_vectors]*norm_imag[kk])/norm;

            fsum_imag[kk + d*num_vectors] = (fsum_imag[kk + d*num_vectors]*norm_real[kk]-old_fsum_real*norm_imag[kk])/norm;

        }
    }
  }



/*
   *   Phase I
   *   multi nth derivative
   *
   */
  void
  finite_sum_with_multi_derivatives_phaseI(double* fsum_real, double* fsum_imag,
						      double* Yinv, double* T, double* zi, double* S,
						      double* deriv_real_in, double* deriv_imag_in, int* n_derivs,
						      int numderivs, int g, int N, int num_vectors)
  {
    // Init
    int offset = 0;
    int nderivs[numderivs];
    double *deriv_real[numderivs];
    double *deriv_imag[numderivs];

    for(int d = 0; d < numderivs; d++) {
      nderivs[d] = n_derivs[d] / g;
      deriv_real[d] = &deriv_real_in[offset];
      deriv_imag[d] = &deriv_imag_in[offset];
      offset += n_derivs[d];

      // Empty
      for (int kk = 0; kk < num_vectors; kk++) {
        fsum_real[kk + d*num_vectors] = 0;
        fsum_imag[kk + d*num_vectors] = 0;
      }
    }

    // Loop over dataset
    for (int kk = 0; kk < num_vectors; kk++)
      {
        double *y = &zi[kk*g];

        int k,j;

        double intshift[g];
        double fracshift[g];
        double sum;

        for (k = 0; k < g; k++) {
	  sum = 0;
	  for (j = 0; j < g; j++) {
	    sum += Yinv[k*g + j] * y[j];
	  }

	  intshift[k] = round(sum);
	  fracshift[k] = sum - intshift[k];
        }

        double npt;
        double dpr[1];
        double dpi[1];
        double* n;

        for(k = 0; k < N; k++) {

	  n = S + k*g;

	  npt = exp(normpart_phaseI(n, T, fracshift, g));

	  for (int d = 0; d < numderivs; d++)
	    {
	      if (n_derivs[d] > 0)
		{
		  dpr[0] = 0;
		  dpi[0] = 0;

		  deriv_prod_phaseI(dpr, dpi, n, intshift, deriv_real[d], deriv_imag[d], nderivs[d], g);

		  fsum_real[kk + d*num_vectors] += dpr[0] * npt;
		  fsum_imag[kk + d*num_vectors] += dpi[0] * npt;

		} else {
          fsum_real[kk + d*num_vectors] += npt;

	      }
	    }
        }


      }
  }


/*
   *   Phase II
   *   Simplified version for purely imaginary Q and purely real z
   *
   */
  void
  finite_sum_with_multi_derivatives_phaseII(double* fsum_real, double* fsum_imag,
						       double* X, double* T, double* zr, double* S,
						       double* deriv_real_in, double* deriv_imag_in, int* n_derivs,
						       int numderivs, int g, int N, int num_vectors)
  {

    int offset = 0;
    int nderivs[numderivs];
    double *deriv_real[numderivs];
    double *deriv_imag[numderivs];
    double dpr[numderivs][1];
    double dpi[numderivs][1];

    for(int d = 0; d < numderivs; d++) {
      nderivs[d] = n_derivs[d] / g;
      deriv_real[d] = &deriv_real_in[offset];
      deriv_imag[d] = &deriv_imag_in[offset];
      offset += n_derivs[d];

      // Empty
      for (int kk = 0; kk < num_vectors; kk++) {
        fsum_real[kk + d*num_vectors] = 0;
        fsum_imag[kk + d*num_vectors] = 0;
      }
    }

    double* n;
    double npt, ept, cpt, spt;

    for(int k = 0; k < N; k++) {
      // the current point in S \subset ZZ^g
      n = S + k*g;
      npt = exp(normpart_phaseII(n, T, g));

      for (int d = 0; d < numderivs; d++) {
        dpr[d][0] = 0;
        dpi[d][0] = 0;
        deriv_prod_phaseII(dpr[d], dpi[d], n, deriv_real[d], deriv_imag[d], nderivs[d], g);
      }

      // Loop over dataset
      for (int kk = 0; kk < num_vectors; kk++)
      {
          double *x = &zr[kk*g];

          // compute the "cosine" and "sine" parts of the summand
          ept = exppart_phaseII(n, X, x, g);
          cpt = npt*cos(ept);
          spt = npt*sin(ept);

          for (int d = 0; d < numderivs; d++)
          {
              if (n_derivs[d] > 0)
              {
                  fsum_real[kk + d*num_vectors] += (dpr[d][0] * cpt - dpi[d][0] * spt);
                  fsum_imag[kk + d*num_vectors] += (dpr[d][0] * spt + dpi[d][0] * cpt);
              }
              else
              {
                  fsum_real[kk + d*num_vectors] += cpt;
                  fsum_imag[kk + d*num_vectors] += spt;

              }
          }
       }
    }


  }






#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __FINITE_SUM_C__ */
