#ifndef _MAP_TO_BIN_H
#define _MAP_TO_BIN_H
/*-----------------------------------------------------------------------------
   Name: MapToBin.h
   Desc: Implements various helper templates for mapping 'values' into 'bins'

   Note #1:  These templates are just an example for generalizing histogram
             behavior to handle other data types and ranges than the original
             TRISH code.  Since the mapper is passed into the TRISH generalize
             methods as a template parameter similar to a functor.  It should
             be easy to write and adapt your own that do any extra or special
             processing required for your own histogram code.   

   Note #2:  For better GPU performance via ILP, we try to process and 
             as many bin values at a time as possible into bin indices
             up to a maximum of 4 values.
             As a result, we can't use true functor like semantics for our
             mapping helper objects.  
             
             So, Instead we assume these mapping function
             have default empty constructors and methods like the following

             void Initiate( valT minVal, valT maxVal, U32 nBins )
             -- Stores necessary range data for mapping
             -- Does any pre-calculation and/or resource allocation

             void Finish()
             -- Does any post-calculation and/or resource cleanup

             // Transform 1 value at a time
             void Transform1( valT inVal, binT & bin1 )     // 32-bit (or 64-bit) version (4 or 8 bytes)
             -- Maps a single value into a single bin indice
             -- Note: only needed for 32-bit and 64-bit data types

             // Transform 2 values at a time
             void Transform2( U32 inVal, binT & bin1, binT & bin2 ); // 16-bit version (2 bytes)
             -- Assumes two 16-bit values stored in single 32-bit storage value
             -- Maps 1 32-bit storage type into two 16-bit value types then into two bin indices for better ILP
             void Transform2( valT inVal1, valT inVal2, binT & bin1, binT & bin2 )  // 32-bit or 64-bit version (4 or 8 bytes)
             -- Maps two input values into two bin indices for better ILP

             // Transform 4
             void Transform4( U32 in1, binT & bin1, binT & bin2, binT & bin3, binT & bin4 ); // 8-bit version (1 byte)
             -- Assumes four 8-bit values stored in a single 32-bit storage value
             -- Maps 1 32-bit storage type into 4 8-bit values then into 4 bin indices for better ILP
             void Transform4( U32 in1, U32 in2, binT & bin1, binT & bin2, binT & bin3, binT & bin4 ); // 16-bit version (2 bytes)
             -- Assumes two 16-bit values stored in two 32-bit storage values
             -- Maps 2 32-bit storage types into 4 16-bit values then into 4 bin indices for better ILP
             void Transform4( valT in1, valT in2, valT in3, valT in4, 
                              binT & bin1, binT & bin2, binT & bin3, binT & bin4 ); // 32-bit or 64-bit version (4 or 8 bytes)
             -- Maps 4 values into 4 bin indices for better ILP


   Note #3:  Here is an explaination of what the various helper template template parameters are used for
      <valT><convT><binT><convFormula><checkMin><checkMax>

<valT> - Underlying value type
    I8 =  8-bit   signed integer
    U8 =  8-bit unsigned integer (BYTE)
   I16 = 16-bit   signed integer
   U16 = 16-bit unsigned integer (WORD)
   I32 = 32-bit   signed integer
   U32 = 32-bit unsigned integer (DWORD)
   I64 = 64-bit   signed integer
   U64 = 64-bit unsigned integer (QWORD)
   F32 = 32-bit  float (single precision real)
   F64 = 64-bit double (double precision real)


<convT> - Underlying conversion type
In other words, an intermediate value that we convert the value type <valT>
into in order to perform the actual mapping operation.

The suggested upscale for conversion to minimize overflow is as follows

Actual | Suggested |
value  | Value     |Conversion Type | Alt. Conversion Type
Type   | Type      |(formula #1)    | (formula #2, #3)
----------------------------------------------------------
  I8   | I32       | F32            | I32
  U8   | U32       | F32            | U32
 I16   | I32       | F32            | I32
 U16   | U32       | F32            | U32
 I32   | I32       | F64            | I64
 U32   | U32       | F64            | U64
 I64*  | I64*      | F64**          | I64**
 U64*  | U64*      | F64**          | F64**
 F32   | F32       | F64            | N/A
 F64*  | U64*      | F64**          | N/A

 * These 64-bit types can result in slower performance on GPUs

** Overflow is a possible problem when converting
   from values to bin indices.  Make sure you verify
   that this will not be a problem for the data ranges
   used in your data set.


<binT> - the resulting type of the underlying bin type

For better performance on the GPU we should set these to 
32-bit integers (U32, I32) which is the intrinsic size of most of the registers.
IE, even though the bins tend to be in the range [0..255] or smaller
we still set the bin type to U32 (32-bit) instead of U8 (8-bit) to
avoid unnecessary bit shift & mask operations which could slow-down
performance.


<convFormula> -  Which formula type to use (1, 2, or 3) <see below>

These 3 formulas below are all derived from a 1-Dimensional linear transformation

To Linear Transform A in range [minA,maxA] into the corresponding B in range [minB,maxB]
   1. Translate minA to origin:  
       A1 = A - minA;         // [minA,maxA] => [0, (maxA-minA)]
   2. Scale A1 to range [0,1]
       A2 = A1 / (maxA-minA); // [0, (maxA-minA)] => [0,1]
   3. Scale A2 to range [0, (maxB-minB)]
       B2 = A2 * (maxB-minB); // [0,1] => [0, (maxB-minB)]
   4. Translate origin to minB:
       B1 = B2 + minB;        // [0, (maxB-minB)] => [minB, maxB]
Note: This assumes that both range A and range B have a non-zero width.

Put it all together
                     wB
    B = (A - minA) * -- + minB,  where width wB = [maxB-minB] and wA = [maxA-minA]
                     wA

Since, We are converting from value range [min,max] to integer bin range [0,n-1], we simplify to

                             nBins
    Bin = (val - minVal) * ---------  
                           (max-min)

However, for integer data the width is actually (max-min+1) instead of (max-min)


Formula #1 - Use for intermediate conversion types of type real (IE F32, F64)
   
 For integer value data types (I8,U8,I16,U16,I32,U32,I64, or U64)
 The formula is 

   Bin = [val - (min - 0.5)] * n/(max-min+1)  

 For better performance, we can precalculate some of the values
 and simplify the formula to:

   Bin = (val - Mu) * Alpha;      // Map (per value)
   where Mu = min - 0.5;          // Pre-calculate once
     and Alpha = n/(max-min+1);   // Pre-calculate once

 For floating value data types (F32 or F64)
 The formula is

   Bin = [val - min] * n/(max-min)

 Which we can precalculate and simplify to:

   Bin = (val - Mu) * Alpha;     // Map (per value)
   where Mu = min;               // Pre-calculate once
     and Alpha = n/(max-min);    // Pre-calculate once

 Pro #1: Faster 
      1. Less Instructions:  fewer arithmetic operations than other mapping formulas
      2. Avoid Divisions:    No divisions during mapping operation
      3. Improved ILP:       Uses floating point cores (FPUs) in parallel to integer cores (ALUs)
 Pro #2: Large integer values resulting in Overflow is much less likely

 Con #1: Floating Point Error (Underflow/Overflow)
      Floating point arithmetic may cause results to be slightly wrong due to round-off error
      in the original data or truncation errors during the intermediate calculations.  This in turn 
      may result in values that are slightly under min (Underflow) or slightly over max (Overflow) which 
      could result in invalid bin values (potentially crashing the code due to out of range array accesses).
      We ignore this problem in our current code but this could be overcome by in a couple of ways...
        Clamping: Clamp final bin values to range [0,nBins-1] at the cost of extra code 
                  and slower performance due to the clamping.
        Padding: Make sure we have room for 'nBins+2' bins and add one to the computed bin after mapping but
                 before returning the results out of the transform*() methods.
                     finalBin = bin + 1;
                 This shifts the results [0,nBins-1] into the range [1,nBins]
            Then the extra two padded bins (bin [0]) and (bin[nBins+1]) can catch these rare underflow/overflow results.
            We would then need to add extra code outside the mapper to cleanup the histogram array after 
            binning all the data values.
            Cleanup Code:
                A. Recover Underflow:  add bin[0] into bin[1]
                B. Recover Overflow:   add bin[nBins+1] into bin[nBins]
                C. Shift:  shifts bin[1,nBins] back into bin[0,nBins-1]


Formula #2:  Use for intermediate conversion types of type integer (I32, U32, I64, U64)

Assumes underlying value data types <valT> are integers (I8,U8,I16,U16,I32,U32,I64, or U64)

      [n*(val - min)] 
Bin = ---------------      // Note: We deliberately drop the 0.5 from formula #1
        (max-min+1)

Simplifies slightly to

Bin = [n*(val-Mu)]/Alpha
where    Mu = min
      Alpha = (max-min+1)

 Pros: None that I can think of

 Con #1: Slower
      1. More arithmetic operations than formula #1 during mapping
      2. Requires a slow Integer division during mapping
      3. No parallel use of FPUs (reduced ILP)
 Con #2: Overflow is a potential problem unless we upscale to a larger data value
         However, we can't upscale from I64, U64
 Con #3: Biases results slightly
      This is due to the drop of the 0.5 from the calculation
      If we have more integer values than bins than by the pigeon hole
      principle some bins will end up with more values than others.
      These bins with extra values will tend to end up at the front of the array 
      due to this bias, whereas with formula #1 or #3 they tend to end up
      in the center of the array.

Formula #3:  Use for intermediate conversion types of type integer (I32, U32, I64, U64)

Assumes underlying value data types <valT> are integers (I8,U8,I16,U16,I32,U32,I64, or U64)

      [n*(2*val - (2*min-1)] 
Bin = ----------------------      // Note: we multiplied by 2/2 to turn 0.5 into an integer
        2*(max-min+1)

 Pros: Doesn't bias results

 Con #1: Slower
      1. Slightly more arithmetic operations than formula #2 during mapping
      2. Requires a slow Integer division during mapping
      3. No parallel use of FPUs (reduced ILP)
 Con #2: Overflow is a potential problem unless we upscale to a larger data value
         However, we can't upscale from I64, U64


<checkMin> - Should we detect and process values that are below our minimum (val < min)
   1 (true)  = Detect and count data values below min (val < min)
       and bin any such values into a special exception bin (nBins+1)
       Pro: Prevents out of range errors, if there are values below our minimum
       Con: Slows down performance (due to extra work to detect and count these below min values)
   0 (false) = Don't do any detection for invalid data values (val < min)
       Pro: Faster performance due to less work
       Con: If our data contains any out of range values (val < min), The resulting
            invalid bin indices (negative or abnormally large) can cause the calling
            code to crash due to out of range array accesses.

<checkMax> - Should we detect and process values that are above our maximum (val > max)
   1 (true)  = Detect and count data values above max (val > max)
       and bin any such values into a special exception bin (nBins+2)
       Pro: Prevents out of range errors, if there are values above our maximum
       Con: Slows down performance (due to extra work to detect and count these above max values)
   0 (false) = Don't do any detection for invalid data values (val > max)
       Pro: Faster performance due to less work
       Con: If our data contains any out of range values (val > max), The resulting
            invalid bin indices (negative or abnormally large) can cause the calling
            code to crash due to out of range array accesses.

Note #4) Gotcha's

Gotcha #1) Due to extreme shared memory usage on the GPU, we can only support up to 256 bin indices

However, for generalized histograms we actually only support 252 bin indices and save
room for 4 special bin indices.  
   Special Bin #1: (value == max) 
      Bin #: (nBins)
      Desc: (For Floating point data only) catch the special case of
      Data Value == Max value (which results in data being put in bin = nBins
      instead of bin = nBins - 1 as expected.
   Special Bin #2: (val < min)
      Bin #: (nBins+1)
      Desc: Only used if (checkMin == 1), detects and counts values such that (val < min)
   Special Bin #3: (val > max)
      Bin #: (nBins+2)
      Desc: Only used if (checkMax == 1), detects and counts values such that (val > max)
   Special Bin #4: (unused)
      User can re-use for special marking flag as needed.

 Gotcha #2) Overflow - overflow can occur when computing indices, 
   For example:  Consider a U8 data type with values in range [0..255]
   and using 252 bins and Formula #3 on a conversion type of U8
   given a value = 255 and min,max = [0,255]

   assume convT = U8
   Bin = [252*(2*255 - (2*0-1))]/2*(255-0+1)   
   Bin = [252*(510 - -1)]/(2*256)  // Overflow Bugs => 510 doesn't fit into 8-bit conversion type, wraps around to 254
                                   // and 256 wraps around to zero
   Bin = [252*(254 + 1)]/(2*0)   // Invalid values propogate through
   Bin = [252*(255)]/0           // Another overflow bug => 252*255 doesn't fit either 64260 => 4
   Bin = [4/0]                   // Divide by zero error, undefined result...

   Now upscale convT = U32 and reuse same example
   Bin = [252*(2*255 - (2*0-1))]/2*(255-0+1)   
   Bin = [252*(510 - -1)]/(2*256) // No overflow
   Bin = [252*511]/512            // No overflow
   Bin = 128772/512               // Note:  128,772 would overflow U16 as well (max value = 65,535)
   Bin = 251                      // No overflow

 Gotcha #3)  When working with integer data, if the # of values is not evenly divisible by the
    number of bins, you will end up with some bins containing more values (and value counts) than others.
    This is not a bug, it's just the way it works.

    For example: if you have 201 unique values in your integer value range which you are binning into
    100 bins.  Then by the "pigeon hole principle" at least one bin will end up with 3 values mapped
    into it. Typically, 99 bins will end up with 2 values each but one bin will end up with 3 values, 
    IE (201/100 = 2 + remainder 1).  If the formula is not biased (#1 & #3), the index of the 3-value 
    bin will probably be the 49th or 50th bin (IE the center of the bin array).  If the formula is 
    biased (#2), the index of the 3-value bin will probably be the 0th bin (IE the front).


Note #5) Examples:

Example #1: 

Say, we know our values are stored 8-bit bytes (U8's) (4 bytes per 32-bit word)
and we want to build a 100 bin histogram on values in the range [20,220]
  minVal = 20
  maxVal = 220  (Range = [20,220] => 201 values)
  nBins  = 100  (Range = [0,99] => 100 bins)

Note: Since we have 201 integer values and 100 bins by the pigeon hole principle at
least one bin will end up with counts for 3 integer values (most bins will end up
with counts for 2 integer values)

Declare the mapper as follows

typedef MAP_INT_B1<U32,F32,U32,1,1,1> myMapper;

where
   valT        = U32 (upscale for better performance on GPU despite underlying data being U8)
   convT       = F32 (We intend to use Formula #1 for faster performance)
   binT        = U32 (upscale for better performance on GPU despite bin indices fitting into a byte = U8)
   convFormula = 1 (we intend to use formula #1 for faster performance)
   checkMin    = 1 (we know that some of our data values are less than 20 and we want to detect and count them)
      -- We also know that these will be stored in the special bin 101 (nBins+1 = 100+1 = 101)
   checkMax    = 1 (we know that some of our data values are greater than 220)

Invoke the count kernel with the proper (min,max,nBins) values

   U32 minVal = 20u;
   U32 maxVal = 220u;
   U32 nBins  = 100u;
   K1_Count
      <..., myMapper, ...>                // Template parameters
      <<<...>>>                           // CTA parameters
      ( ..., minVal, maxVal, numBins );   // Function Parameters

 The Count Kernel will invoke our mapper in the following ways

   // Create a GPU instance of the mapper helper object
   mapT mapper;      
   ...
   
   // Initialize the mapper with [minVal,maxVal] and nBins
   mapper.Initiate( minVal, maxVal, nBins ); 
      // This Initiate method is only called once per CTA block
      // Our method Pre-calculates 'Mu' and 'Alpha' for formula #1, #2, or #3
      // Your method could do whatever it needs to
   ...

      // Tranform values into bin indices
   mapper.Transform4( inVal, bin1, bin2, bin3, bin4 );
      // Transform 32-bit 'inVal' into 4 8-bit values (stored in 32-bit registers)
      // Then transforms those 4 8-bit values into F32's (for formula #1)
      // Transforms those 4 F32's into bin indices (using formula #1)  Bin = (val - mu) * Alpha;
      // Converts those 4 F32's back into 4 U32's for bin indices
      // returns the 4 bin indices
      // This function is invoked once per 32-bit storage value (every 4 bytes)

   ...
   mapper.Finish();  // Post-processing and cleanup
      // Our method does nothing currently
      // Your method could post-process (compute sums) or cleanup allocated resources
   
Example #2: 

   Say we want to work with 32-bit integer data as I32's in the range [-10000,+70000]
   using 200 bins.  Say further, we know that all the data values are inside this range
   so we don't need to prevent out of range accesses.

Declare the mapper as follows

typedef MAP_INT_B1<I32,F32,U32,1,1,1> myMapper;
   valT        = I32 
   convT       = F32 (We don't need to use F64 as the value range is small enough to not overflow)
   binT        = U32 (upscale for better performance on GPU despite bin indices fitting into a byte = U8)
   convFormula = 1 (we intend to use formula #1 for faster performance)
   checkMin    = 0 (safe to avoid check [val < min], faster performance)
   checkMax    = 1 (safe to avoid check [val > max], faster performance)

The kernel behavior is similar to above with the exception that the mapper needs to call a different
transform function (IE each value is a 32-bit value, no need to map from storage first)

   mapper.Transform4( val1, val2, val3, val4,      // 4 input values
                      bint1, bin2, bin3, bin4 );   // 4 output bin indices

   Disclaimer:
      This software is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
  Includes
-----------------------------------------------------------------------------*/

#include "BaseDefs.h"


/*-----------------------------------------------------------------------------
  Mapping Templates
-----------------------------------------------------------------------------*/

/*-----------------------------------------------
  Name:  MapToBin
  Desc:  Map values to bins
  Note:  Assumes Underlying value types
         are 1 byte integers (U8,I8)
-----------------------------------------------*/

template < 
           typename valT,                // Underlying value type
           typename convT       = F32,   // Conversion Type
           typename binT        = U32,   // Underlying bin type
           U32      convFormula = 1u,    // Underlying Formula to use (#1,#2,#3)
           U32      checkMin    = 1u,    // Range check values outside of [min,max]
           U32      checkMax    = 1u     //       ditto
         >  
struct MapToBin
{
   // Fields
   valT  m_min;
   valT  m_max;
   binT  m_nBins;
   convT m_Mu;
   convT m_Alpha;

   __host__ __device__ __forceinline__
   void SetValues( valT minV, valT maxV, binT nBins )
   {
      if (minV > maxV) 
      {
         m_min = maxV;
         m_max = minV;
      }
      else if (maxV > minV)
      {
         m_min = minV;
         m_max = maxV;
      }
      else
      {
         // Error min and max are equal, what should we do
      }

      m_nBins = nBins;
      if (m_nBins < 2) 
      {
         m_nBins = 2;
      }
      if (m_nBins >= 256)
      {
         m_nBins = 256;
      }
   }


	__host__ __device__ __forceinline__
   void ComputeAlpha()
      {
         if (1u == convFormula)
         {
            // Convert using floating point (F32 or F64)
            //    Mu = min - 0.5
            // Alpha = n/(max-min+1)
            convT wA = (convT)(m_max - m_min + valT(1));
            convT wB = (convT)m_nBins;
            m_Mu    = convT(m_min)-convT(0.5);
            m_Alpha = wB/wA;
         }
         if (2u == convFormula)
         {
            // Convert using integers (I32, U32, I64, or U64)
            // Just ignore (Drop) the 0.5
            //    Mu = min;      
            // Alpha = 2*(max-min+1)
            m_Mu    = convT(m_min);
            m_Alpha = (convT)(m_max - m_min + valT(1));
         }
         if (3u == convFormula)
         {
            // Convert using integers (I32, U32, I64, or U64)
            // Convert 0.5 to an integer by multiplying through by 2/2
            //    Mu = 2*min - 1    
            // Alpha = 2*(max-min+1)
            m_Mu    = (convT(2) * convT(m_min)) - convT(1);                    
            m_Alpha = convT(2) * ((convT)(m_max) - convT(m_min) + convT(1));   
         }
      }

	__host__ __device__ __forceinline__
   void SetRange( valT minV, valT maxV, binT numBins )
      {
         SetValues( minV, maxV, numBins );
         ComputeAlpha();
      }

	__host__ __device__ __forceinline__
   MapToBin()
      {
      }

   // Initiate - Setup here
	__host__ __device__ __forceinline__
   void Initiate( valT minV, valT maxV, binT numBins )
      {
         SetRange( minV, maxV, numBins );
      }

   // Finish - Cleanup here
	__host__ __device__ __forceinline__
   void Finish()
      {
      }

   // Transform - Convert four values into four bins
	__host__ __device__ __forceinline__
	void Transform4
      ( 
         binT & bin1, binT & bin2, binT & bin3, binT & bin4, // OUT - bins   [1..4]
         valT   val1, valT   val2, valT   val3, valT   val4  // IN  - values [1..4] to bin
      )
		{ 
         //-----
         // Map 'Values' into 'Bins'
         //-----

         binT s1, s2, s3, s4;
         if (1u == convFormula)
         {
            //-
            // Formula #1:  Bin = (val - (min-0.5)) * [n/(max-min+1)]
            //-

            // Convert values to intermediate type (convT)
            convT t1, t2, t3, t4;
            t1 = (convT)val1;
            t2 = (convT)val2;
            t3 = (convT)val3;
            t4 = (convT)val4;

            // Subtract Mu (min - 0.5)
            t1 = t1 - m_Mu;
            t2 = t2 - m_Mu;
            t3 = t3 - m_Mu;
            t4 = t4 - m_Mu;

            // Multiply by Alpha [n/(max-min+1)]
            t1 = t1 * m_Alpha;
            t2 = t2 * m_Alpha;
            t3 = t3 * m_Alpha;
            t4 = t4 * m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
            s2 = (binT)t2;
            s3 = (binT)t3;
            s4 = (binT)t4;
         }

         if (2u == convFormula)
         {
            //-
            // Formula #2:  Bin = [nBins*(val - min)]/(max-min+1)
            //-

            // Convert values to intermediate type (convT)
            convT t1, t2, t3, t4, n1;
            t1 = (convT)val1;
            t2 = (convT)val2;
            t3 = (convT)val3;
            t4 = (convT)val4;
            n1 = (convT)m_nBins;

            // Subtract Mu (min)
            t1 = t1 - m_Mu;
            t2 = t2 - m_Mu;
            t3 = t3 - m_Mu;
            t4 = t4 - m_Mu;

            // Multiply by nBins
            t1 = t1 * n1;
            t2 = t2 * n1;
            t3 = t3 * n1;
            t4 = t4 * n1;

            // Divide by Alpha (max-min+1)
            t1 = t1 / m_Alpha;
            t2 = t2 / m_Alpha;
            t3 = t3 / m_Alpha;
            t4 = t4 / m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
            s2 = (binT)t2;
            s3 = (binT)t3;
            s4 = (binT)t4;
         }

         if (3u == convFormula)
         {
            //-
            // Formula #3:  Bin = [nBins*(2*val - (2*min-1))]/[2*(max-min+1)]
            //-

            // Convert values to intermediate type (convT)
            convT t1, t2, t3, t4, n;
            t1 = (convT)val1;
            t2 = (convT)val2;
            t3 = (convT)val3;
            t4 = (convT)val4;
            n = (convT)m_nBins;

            // Multiply value by 2 (2*val)
            t1 = convT(2) * t1;
            t2 = convT(2) * t2;
            t3 = convT(2) * t3;
            t4 = convT(2) * t4;

            // Subtract Mu (2*min - 1)
            t1 = t1 - m_Mu;
            t2 = t2 - m_Mu;
            t3 = t3 - m_Mu;
            t4 = t4 - m_Mu;

            // Multiply by nBins
            t1 = t1 * n;
            t2 = t2 * n;
            t3 = t3 * n;
            t4 = t4 * n;

            // Divide by alpha [2*(max-min+1)]
            t1 = t1 / m_Alpha;
            t2 = t2 / m_Alpha;
            t3 = t3 / m_Alpha;
            t4 = t4 / m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
            s2 = (binT)t2;
            s3 = (binT)t3;
            s4 = (binT)t4;
         }


         //-----
         // check value against [min,max] to prevent out of range accesses
         //-----

         if (0u < checkMin)
         {
            // Check for 'values' below 'min'
            binT n1 = m_nBins + binT(1);  // Special bin (underflow counts)

            U32 test1, test2, test3, test4;
            test1 = (val1 < m_min);
            test2 = (val2 < m_min);
            test3 = (val3 < m_min);
            test4 = (val4 < m_min);

            s1 = (test1 ? n1 : s1);
            s2 = (test2 ? n1 : s2);
            s3 = (test3 ? n1 : s3);
            s4 = (test4 ? n1 : s4);
         }

         if (0u < checkMax)
         {
            // Check for 'values' above 'max'
            binT n2 = m_nBins + binT(2);  // Special bin (overflow counts)

            U32 test1, test2, test3, test4;
            test1 = (val1 > m_max);
            test2 = (val2 > m_max);
            test3 = (val3 > m_max);
            test4 = (val4 > m_max);

            s1 = (test1 ? n2 : s1);
            s2 = (test2 ? n2 : s2);
            s3 = (test3 ? n2 : s3);
            s4 = (test4 ? n2 : s4);
         }

         // Return bin indices
         bin1 = s1;
         bin2 = s2;
         bin3 = s3;
         bin4 = s4;
		}


   // Transform - Convert three values into three bins
	__host__ __device__ __forceinline__
	void Transform3
      ( 
         binT & bin1, binT & bin2, binT & bin3, // OUT - bins   [1..3]
         valT   val1, valT   val2, valT   val3  // IN  - values [1..3] to bin
      )
		{ 
         //-----
         // Map 'Values' into 'Bins'
         //-----

         binT s1, s2, s3;
         if (1u == convFormula)
         {
            //-
            // Formula #1:  Bin = (val - (min-0.5)) * [n/(max-min+1)]
            //-

            // Convert values to intermediate type (convT)
            convT t1, t2, t3;
            t1 = (convT)val1;
            t2 = (convT)val2;
            t3 = (convT)val3;

            // Subtract Mu (min - 0.5)
            t1 = t1 - m_Mu;
            t2 = t2 - m_Mu;
            t3 = t3 - m_Mu;

            // Multiply by Alpha [n/(max-min+1)]
            t1 = t1 * m_Alpha;
            t2 = t2 * m_Alpha;
            t3 = t3 * m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
            s2 = (binT)t2;
            s3 = (binT)t3;
         }

         if (2u == convFormula)
         {
            //-
            // Formula #2:  Bin = [nBins*(val - min)]/(max-min+1)
            //-

            // Convert values to intermediate type (convT)
            convT t1, t2, t3, n1;
            t1 = (convT)val1;
            t2 = (convT)val2;
            t3 = (convT)val3;
            n1 = (convT)m_nBins;

            // Subtract Mu (min)
            t1 = t1 - m_Mu;
            t2 = t2 - m_Mu;
            t3 = t3 - m_Mu;

            // Multiply by nBins
            t1 = t1 * n1;
            t2 = t2 * n1;
            t3 = t3 * n1;

            // Divide by Alpha (max-min+1)
            t1 = t1 / m_Alpha;
            t2 = t2 / m_Alpha;
            t3 = t3 / m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
            s2 = (binT)t2;
            s3 = (binT)t3;
         }

         if (3u == convFormula)
         {
            //-
            // Formula #3:  Bin = [nBins*(2*val - (2*min-1))]/[2*(max-min+1)]
            //-

            // Convert values to intermediate type (convT)
            convT t1, t2, t3, n;
            t1 = (convT)val1;
            t2 = (convT)val2;
            t3 = (convT)val3;
            n = (convT)m_nBins;

            // Multiply value by 2 (2*val)
            t1 = convT(2) * t1;
            t2 = convT(2) * t2;
            t3 = convT(2) * t3;

            // Subtract Mu (2*min - 1)
            t1 = t1 - m_Mu;
            t2 = t2 - m_Mu;
            t3 = t3 - m_Mu;

            // Multiply by nBins
            t1 = t1 * n;
            t2 = t2 * n;
            t3 = t3 * n;

            // Divide by alpha [2*(max-min+1)]
            t1 = t1 / m_Alpha;
            t2 = t2 / m_Alpha;
            t3 = t3 / m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
            s2 = (binT)t2;
            s3 = (binT)t3;
         }


         //-----
         // check value against [min,max] to prevent out of range accesses
         //-----

         if (0u < checkMin)
         {
            // Check for 'values' below 'min'
            binT n1 = m_nBins + binT(1);  // Special bin (underflow counts)

            U32 test1, test2, test3;
            test1 = (val1 < m_min);
            test2 = (val2 < m_min);
            test3 = (val3 < m_min);

            s1 = (test1 ? n1 : s1);
            s2 = (test2 ? n1 : s2);
            s3 = (test3 ? n1 : s3);
         }

         if (0u < checkMax)
         {
            // Check for 'values' above 'max'
            binT n2 = m_nBins + binT(2);  // Special bin (overflow counts)

            U32 test1, test2, test3;
            test1 = (val1 > m_max);
            test2 = (val2 > m_max);
            test3 = (val3 > m_max);

            s1 = (test1 ? n2 : s1);
            s2 = (test2 ? n2 : s2);
            s3 = (test3 ? n2 : s3);
         }

         // Return bin indices
         bin1 = s1;
         bin2 = s2;
         bin3 = s3;
		}

   // Transform - Convert two values into two bins
	__host__ __device__ __forceinline__
	void Transform2
      ( 
         binT & bin1, binT & bin2, // OUT - bins   [1..2]
         valT   val1, valT   val2  // IN  - values [1..2] to bin
      )
		{ 
         //-----
         // Map 'Values' into 'Bins'
         //-----

         binT s1, s2;
         if (1u == convFormula)
         {
            //-
            // Formula #1:  Bin = (val - (min-0.5)) * [n/(max-min+1)]
            //-

            // Convert values to intermediate type (convT)
            convT t1, t2;
            t1 = (convT)val1;
            t2 = (convT)val2;

            // Subtract Mu (min - 0.5)
            t1 = t1 - m_Mu;
            t2 = t2 - m_Mu;

            // Multiply by Alpha [n/(max-min+1)]
            t1 = t1 * m_Alpha;
            t2 = t2 * m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
            s2 = (binT)t2;
         }

         if (2u == convFormula)
         {
            //-
            // Formula #2:  Bin = [nBins*(val - min)]/(max-min+1)
            //-

            // Convert values to intermediate type (convT)
            convT t1, t2, n1;
            t1 = (convT)val1;
            t2 = (convT)val2;
            n1 = (convT)m_nBins;

            // Subtract Mu (min)
            t1 = t1 - m_Mu;
            t2 = t2 - m_Mu;

            // Multiply by nBins
            t1 = t1 * n1;
            t2 = t2 * n1;

            // Divide by Alpha (max-min+1)
            t1 = t1 / m_Alpha;
            t2 = t2 / m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
            s2 = (binT)t2;
         }

         if (3u == convFormula)
         {
            //-
            // Formula #3:  Bin = [nBins*(2*val - (2*min-1))]/[2*(max-min+1)]
            //-

            // Convert values to intermediate type (convT)
            convT t1, t2, n;
            t1 = (convT)val1;
            t2 = (convT)val2;
            n = (convT)m_nBins;

            // Multiply value by 2 (2*val)
            t1 = convT(2) * t1;
            t2 = convT(2) * t2;

            // Subtract Mu (2*min - 1)
            t1 = t1 - m_Mu;
            t2 = t2 - m_Mu;

            // Multiply by nBins
            t1 = t1 * n;
            t2 = t2 * n;

            // Divide by alpha [2*(max-min+1)]
            t1 = t1 / m_Alpha;
            t2 = t2 / m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
            s2 = (binT)t2;
         }


         //-----
         // check value against [min,max] to prevent out of range accesses
         //-----

         if (0u < checkMin)
         {
            // Check for 'values' below 'min'
            binT n1 = m_nBins + binT(1);  // Special bin (underflow counts)

            U32 test1, test2;
            test1 = (val1 < m_min);
            test2 = (val2 < m_min);

            s1 = (test1 ? n1 : s1);
            s2 = (test2 ? n1 : s2);
         }

         if (0u < checkMax)
         {
            // Check for 'values' above 'max'
            binT n2 = m_nBins + binT(2);  // Special bin (overflow counts)

            U32 test1, test2;
            test1 = (val1 > m_max);
            test2 = (val2 > m_max);

            s1 = (test1 ? n2 : s1);
            s2 = (test2 ? n2 : s2);
         }

         // Return bin indices
         bin1 = s1;
         bin2 = s2;
		}


   // Transform - Convert two values into two bins
	__host__ __device__ __forceinline__
	void Transform1
      ( 
         binT & bin1, // OUT - bins   [1..3]
         valT   val1  // IN  - values [1..3] to bin
      )
		{ 
         //-----
         // Map 'Values' into 'Bins'
         //-----

         binT s1;
         if (1u == convFormula)
         {
            //-
            // Formula #1:  Bin = (val - (min-0.5)) * [n/(max-min+1)]
            //-

            // Convert values to intermediate type (convT)
            convT t1;
            t1 = (convT)val1;

            // Subtract Mu (min - 0.5)
            t1 = t1 - m_Mu;

            // Multiply by Alpha [n/(max-min+1)]
            t1 = t1 * m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
         }

         if (2u == convFormula)
         {
            //-
            // Formula #2:  Bin = [nBins*(val - min)]/(max-min+1)
            //-

            // Convert values to intermediate type (convT)
            convT t1, n1;
            t1 = (convT)val1;
            n1 = (convT)m_nBins;

            // Subtract Mu (min)
            t1 = t1 - m_Mu;

            // Multiply by nBins
            t1 = t1 * n1;

            // Divide by Alpha (max-min+1)
            t1 = t1 / m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
         }

         if (3u == convFormula)
         {
            //-
            // Formula #3:  Bin = [nBins*(2*val - (2*min-1))]/[2*(max-min+1)]
            //-

            // Convert values to intermediate type (convT)
            convT t1, n;
            t1 = (convT)val1;
            n = (convT)m_nBins;

            // Multiply value by 2 (2*val)
            t1 = convT(2) * t1;

            // Subtract Mu (2*min - 1)
            t1 = t1 - m_Mu;

            // Multiply by nBins
            t1 = t1 * n;

            // Divide by alpha [2*(max-min+1)]
            t1 = t1 / m_Alpha;

            // Convert to bin type
            s1 = (binT)t1;
         }


         //-----
         // check value against [min,max] to prevent out of range accesses
         //-----

         if (0u < checkMin)
         {
            // Check for 'values' below 'min'
            binT n1 = m_nBins + binT(1);  // Special bin (underflow counts)

            U32 test1;
            test1 = (val1 < m_min);

            s1 = (test1 ? n1 : s1);
         }

         if (0u < checkMax)
         {
            // Check for 'values' above 'max'
            binT n2 = m_nBins + binT(2);  // Special bin (overflow counts)

            U32 test1;
            test1 = (val1 > m_max);

            s1 = (test1 ? n2 : s1);
         }

         // Return bin indices
         bin1 = s1;
		}

}; // End MapToBin

#endif // _MAP_TO_BIN_H