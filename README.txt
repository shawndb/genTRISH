Overview:
This is the directory for the 'genTRISH' sample application.

This demo is based on the original "NVIDIA" CUDA 4.0 "histogram" sample
application.  The original sample uses Podlozhnyuk's histogram method.
This demo adds support for a new histogram method called "TRISH".


TRISH Method: 
The original version of TRISH only supported 8-bit data (bytes) using
exactly 256 bins.  This allowed faster GPU code similar to the following 
CPU snippet...

// Input: an input array named A containing 'n' 8-bit bytes
// Output: an output array named 'counts' containing 256 32-bit counts
// Histogram method
for (i=0; i< n;i++) 
{
  counts[A[i]]++;
}

TRISH uses Thread level parallelism (Occupancy = 12.5% = 192/1536), 
Instruction Level parallelism (loop unrolling & batching), Vector 
Parallelism (applying arithmetic operations on byte pairs instead of 
4 individual) bytes to speed up overall performance.

More Importantly, TRISH is deterministic and lock-free. IE it doesn't 
use atomics.  This means that you get similar performance regardless 
of the underlying data distribution.  Podlozhnyuk's original histogram 
method's performance will vary depending on how many thread collisions 
the  underlying data causes when binning & counting.


Generalized TRISH:
This demo contains a generalized version of TRISH that supports other data
types (I8, I16, I32, I64, U8, U16, U32, I64, F32, F64)
   I8, I16, I32, I64 => 8-bit, 16-bit, 32-bit, and 64-bit signed integers.
   U8, U16, U32, U64 => 8-bit, 16-bit, 32-bit, and 64-bit unsigned integers.
   F32, F64 => single precision (32-bit) and double precision (64-bit) reals.

It also supports variable number of bins in the range [1..252].  Theoretically,
it could support up to 256 bins but we reserve the last 4 bins for marking special
cases.


Mapper Object:
The code uses a special transform object to convert from "values" to "bins".
This is similar in spirit to a "functor" object.  But, since we try to 
process 4 items at a time for better ILP performance, the functor approach can't capture
all the different inputs and outputs and transform methods.  Take a look at the include file MapToBin.h to see how this works.

Basically a Mapper template contains the following Methods
    A default constructor that does nothing
    ::Initiate( minVal, maxVal, nBins ) - initates the mapper from inputs
    ::Finish() - Opportunity to cleanup
    ::Transform4() - transform 4 values to 4 bins for better ILP
    ::Transform3() - transform 3 values to 3 bins (to catch any left over values)
    ::Transform2() - transform 2 values to 2 bins (to catch any left over values)
    ::Transform1() - transform 1 value to 1 bin (to catch any left over values)


Transform Formula:
We adapted a 1D linear transform from one range to another.
Resulting in the following formula (for F32, F64).

Bin = (val - minVal) * [nBins/(maxVal-minVal)];

For integer data (I8,I16,I32,U8,U16,U32) we need to change the formula to

Bin = (val - (minVal-0.5)) * [nBins/(maxVal-minVal+1)];


For better performance, we use F32 (or F64) as our intermediate 
conversion data type and simplify the above formulas to

Bin = (val - Mu) * Alpha.

Where  Mu = (val-MinVal) for floats or (val-(minVal-0.5) for integers.
and Alpha = n/(max-min) for floats or n/(max-min+1) for integers.


Design Issues:
1. Values per Storage Value:
The current design still feels awkward as I don't have a clean interface
as the code currently has to differentiate between
   1-byte data types (4 byte values contained per 32-bit storage value)
   2-byte data types (2 word values contained per 32-bit storage value)
   4-byte & 8-byte data types (1 original value = 1 storage value)

2. Formulas:
I current have hard-coded 3 formulas 
Formula #1 (described above) using floats for the actual conversion
Formula #2 & #3 described in MapToBin.h that use integers for the actual
conversion.
   - Unfortunately, these methods are slower (require integer divide)
     and are more likely to overflow the input range, so I don't use them
     currently.

3. Overflow:  Converting from values to bins may overflow the conversion
   type resulting in 'out of range' array accesses which will crash the code.  
   Make sure you understand your input data range and conversion ranges so
   that you can prevent this from happening.

4. Clamping: The mapper objects also support clamping values outside the
   specified [min,max] range and counting those underflow and overflow values
   in extra bins (nBins+1) and (nBins+2) respectively.  This extra clamping
   logic hurts performance as we need to do extra tests but helps prevent
   crashes due to input values that are outside the expected range.



Misc Notes:


1.)  The code was written and tested using the following environment
     OS:  Windows 7, SP1
     CUDA:  4.0
     IDE: Microsoft Visual Studio 2008
          Microsoft Visual Studio 2010
     Code Generation:  WIN32, WIN64


2.)  Furthermore, the code was written, developed, and tested
     under the assumption that it is being compiled as 
     part of the demo projects in the NVidia Computing SDK 4.0.  
     IE that we created and stored this directory parallel to all 
     the other sample projects in the CUDA SDK.  
     The Computing 4.0 SDK was found in the following directory 
     on my machine.

c:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK 4.0\C\SRC

     So, this modified "histogram" application should be stored at

...\...\NVIDIA GPU Computing SDK 4.0\c\SRC\demoTRISH

     To build properly...


3.)  List of files.  Here is a list of files that I use to actually
     build my "histogram" demo application.

   // Include files
   BaseDefs.h
   Extract.h
   histogram_common.h
   MapToBin.h
   Platform.h
   TRISH_Traits.h
   
   // Source Files
   histogram256.cu
   histogram64.cu
   histogram_gold.cpp
   histTRISH_Gen.cu
   histTRISH_gold.cu
   main.cpp

The most important files are 
   Extract.h	// Extracts 4 bytes or 2 words from a 32-bit storage value 
   MapToBin.h   // Converts [1..4] values into [1..4] bins
   histTRISH_Gen.cu // GPU kernels for implementing Generalized TRISH


4.) To get your application to run properly, You may need to 
    include some CUDA DLL's in the same directory as your application 
    or in your windows system directory

    c:\windows\system32 
or 
    c:\windows\SysWOW64

    This might include the following DLL's
       // For 32-bit apps
       cutil32.dll		
       cudart32_40_17.dll

       // For 64-bit apps	
       cutil64.dll      
       cudart64_40_17.dll

    The cutil*.dll files are part of the CUDA 4.0 GPU SDK
    The cudart*.dll files are of the CUDA 4.0 Toolkit
 

6.) Limitations:

** CUDA Enviroment.
    CUDA 3.2 - Should work OK, but I have not tested it.
    CUDA 4.0 - Generates good kernel code (36 registers for Count Kernel)
    CUDA 4.1 - Not so Good, 4.1 Causes the TRISH kernels to blow up 
               Both kernels end up using 63 registers + a large # of 
               register spills.  This negatively impacts performance, 
               for about a 20% loss, caused by dealing with the extra 
               overhead of "Local" memory load/stores for dealing with 
               the register spills.
    CUDA 4.2 - Not yet tested
    CUDA 5.0 - Not yet available at the time I wrote this document...

** Alignment: Assumes input data is actually 32-bit elements (not true byte data)
   -- IE Input data aligned to 32-bit boundaries and nElems is measured in
      32-bit elements (4 bytes) not in actually bytes

** Work Per thread 
   4 Bytes:  K-value in range [1..63], best K = 31 (or 63)
   2 Words:  K-value in range [1..127], best K = 63
      Note:  K > 104 slows down performance probably because the code is now
             to large to fit into the code cache properly...
   1 value (32 or 64-bit):  K-Value in range [1..255], best K = 15 (or 127)
      Note:  I have gone to the effort to create code to support K > 127 yet...  
             so K is actually limited to [1..127] for now...

   If you want to pick a different 'k' value, change the following line of code
   in the appropriate CPU wrapper function.

const uint K1_Length    = 31u;		//  31 = Work Per thread (loop unrolling)


** BlockSize: Maximum Threads per Block is 64 (for good performance)
   Shared Memory Usage = 16 KB (64 threads * 64 lanes * 4 bytes) 
   On Fermi cards, this implies 3 concurrent blocks per SM (48KB/16KB)

** GridSize: Best performance is achieved when you specify the Grid Size as 
     #Blocks = #SMs * concurrent blocks per SM
     48 = 16 SMs * 3 concurrent blocks per SM (on GTX 580)
     45 = 15 SMs * 3 concurrent blocks per SM (on GTX 480)
     42 = 14 SMs * 3 concurrent blocks per SM (on Telsa M2050)
     12 =  4 SMs * 3 concurrent blocks per SM (on GTX 560M)

So, don't forget to adapt this for your hardware.
change the corresponding NUM_GPU_SMS #define in histTRISH_Gen.cu 
to the correct value for your hardware.

If you set this value incorrectly for your hardware, 
you will probably see a significant slowdown in performance.




/*-----------------
  Local Defines
-----------------*/

// GTX 560M
//#define NUM_GPU_SMs (4u)

// TESLA 2050 (2070)
//#define NUM_GPU_SMs (14u)

// GTX 480
#define NUM_GPU_SMs (15u)

// GTX 580
//#define NUM_GPU_SMs (16u)
 