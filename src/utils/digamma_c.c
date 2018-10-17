/* digamma.c
 *
 * Mark Johnson, 2nd September 2007
 *
 * Computes the Ψ(x) or digamma function, i.e., the derivative of the 
 * log gamma function, using a series expansion.
 *
 * Warning:  I'm not a numerical analyst, so I may have made errors here!
 *
 * The parameters of the series were computed using the Maple symbolic
 * algebra program as follows:
 *
 * series(Psi(x+1/2), x=infinity, 21);
 *
 * which produces:
 *
 *  ln(x)+1/(24*x^2)-7/960/x^4+31/8064/x^6-127/30720/x^8+511/67584/x^10-1414477/67092480/x^12+8191/98304/x^14-118518239/267386880/x^16+5749691557/1882718208/x^18-91546277357/3460300800/x^20+O(1/(x^21)) 
 *
 * It looks as if the terms in this expansion *diverge* as the powers
 * get larger.  However, for large x, the x^-n term will dominate.
 *
 * I used Maple to examine the difference between this series and
 * Digamma(x+1/2) over the range 7 < x < 20, and discovered that the
 * difference is less that 1e-8 if the terms up to x^-8 are included.
 * This determined the power used in the code here.  Of course,
 * Maple uses some kind of approximation to calculate Digamma,
 * so all I've really done here is find the smallest power that produces
 * the Maple approximation; still, that should be good enough for our
 * purposes.
 *
 * This expansion is accurate for x > 7; we use the recurrence 
 *
 * digamma(x) = digamma(x+1) - 1/x
 *
 * to make x larger than 7.
 */

#include <assert.h>
#include <math.h>

double digamma(double x) {
  double result = 0, xx, xx2, xx4;
  assert(x > 0);
  for ( ; x < 7; ++x)
    result -= 1/x;
  x -= 1.0/2.0;
  xx = 1.0/x;
  xx2 = xx*xx;
  xx4 = xx2*xx2;
  result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
  return result;
}

#ifdef TEST
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  double x;
  for (x = 0.1; x < 10; x += 0.1)
    printf("digamma(%g) = %g, exp(digamma(%g)) = %g\n", x, digamma(x), x, exp(digamma(x)));
  exit(EXIT_SUCCESS);
}

#endif /* TEST */