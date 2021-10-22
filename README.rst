.. -*- mode: rst -*-

.. image:: https://github.com/CCampJr/Hilbert/actions/workflows/python-testing.yml/badge.svg
	:alt: pytest
	:target: https://github.com/CCampJr/Hilbert/actions/workflows/python-testing.yml

.. image:: https://codecov.io/gh/CCampJr/Hilbert/branch/master/graph/badge.svg?token=WIHgHEUc82
	:alt: Codecov
	:target: https://codecov.io/gh/CCampJr/Hilbert

.. image:: https://img.shields.io/badge/License-NIST%20Public%20Domain-green.svg
    :alt: NIST Public Domain
    :target: https://github.com/CCampJr/Hilbert/blob/master/LICENSE.md

Hilbert - Discrete Hilbert Transform Implementations
============================================================

**Hilbert** is a project that will contain numerous implementations (and 
approximations) of the discrete Hilbert transform.

Currently, this package is a work in progress and should probably **not be used**. 

Currently Implemented
----------------------

-   Discrete Fourier Transform-based

    -   Henrici [1]_
    -   Marple (SciPy and MATLAB's *hilbert* implementation) [2]_
    -   Haar wavelet-based (similar to Zhou-Yang [3]_)

References
~~~~~~~~~~~

.. [1] P. Henrici, Applied and Computational Complex Analysis Vol III 
       (Wiley-Interscience, 1986).
        
.. [2] L. Marple, "Computing the discrete-time “analytic” signal via FFT," 
       IEEE Trans. Signal Process. 47(9), 2600–2603 (1999).

.. [3] C. Zhou, L. Yang, Y. Liu, and Z. Yang, "A novel method for computing 
       the Hilbert transform with Haar multiresolution approximation," J. Comput. 
       Appl. Math. 223(2), 585–597 (2009).

Coming Soon
------------

-   Implementations

    -   B-splines implementation (Bilato)
    -   Haar multiresolution (Zhou-Yang)
    -   Sinc / Whittaker Cardinal
    -   and more!

-   Documentation
-   Jupyter Notebook Examples


Dependencies
------------

Installation
-------------

Usage
------

Citing This Software
---------------------

LICENSE
----------
This software was developed by employees of the National Institute of Standards 
and Technology (NIST), an agency of the Federal Government. Pursuant to 
`title 17 United States Code Section 105 <http://www.copyright.gov/title17/92chap1.html#105>`_, 
works of NIST employees are not subject to copyright protection in the United States and are 
considered to be in the public domain. Permission to freely use, copy, modify, 
and distribute this software and its documentation without fee is hereby granted, 
provided that this notice and disclaimer of warranty appears in all copies.

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER 
EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY 
THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, 
AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY 
WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE. IN NO EVENT SHALL NIST BE LIABLE 
FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR 
CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED 
WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR 
OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR 
OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE 
RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.

Portions of this package include source code edited from the `sklearn's project template`_, which
requires the following notice(s):

.. _sklearn's project template: https://github.com/scikit-learn-contrib/project-template/blob/master/doc/index.rst

Copyright (c) 2016, Vighnesh Birodkar and scikit-learn-contrib contributors
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Contact
-------
Charles H Camp Jr: `charles.camp@nist.gov <mailto:charles.camp@nist.gov>`_