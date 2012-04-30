/*-----------------------------------------------------------------------------
   Name: histTRISH_Gen.cu
   Desc: Implements generic binning histograms on GPU
   
   Disclaimer:
      This software is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
  Includes
-----------------------------------------------------------------------------*/

// System Includes
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA Includes
#include <cutil_inline.h>

// Local Includes
#include "Platform.h"
#include "BaseDefs.h"
#include "TRISH_traits.h"
#include "MapToBin.h"
#include "Extract.h"
#include "histogram_common.h"


/*-----------------------------------------------------------------------------
  Compiler Settings
-----------------------------------------------------------------------------*/

//#define INTERLEAVE 1
//#define INTERLEAVE 2
#define INTERLEAVE 4

//#define TRISH_VERIFY_HISTOGRAM 1
#define TRISH_VERIFY_HISTOGRAM 0


/*-----------------------------------------------------------------------------
  Helper Templates
-----------------------------------------------------------------------------*/



/*-------------------------------------
  Name:  TRISH_VerifyHistogram
-------------------------------------*/

#if 1 == TRISH_VERIFY_HISTOGRAM
// Verify single byte integers (I8, U8)
template < 
           typename valT,     // Underlying value type
           typename mapT      // Mapper Type
         >
__host__
void TRISH_VerifyHistogram_B1
( 
	 U32   nElems,          // IN - number of 32-bit elements to bin & count
	 U32 * d_gpuElems,      // IN - array of elements to bin & count
    U32   numBins,         // IN - number of bins in histogram
	 U32 * d_gpuCounts,     // IN - GPU histogram counts
    valT  minVal,          // IN - [min,max] values for histogram
    valT  maxVal           //      ditto
)
{
   assert( numBins > 0u );
   assert( numBins <= 256u );
   assert( nElems > 0u );

   U32 mem_size_elems  = nElems * sizeof( U32 );
   U32 mem_size_counts = 256u * sizeof( U32 );

   U32 * h_cpuElems  = NULL;
   U32 * h_gpuCounts = NULL;
   U32 * h_cpuCounts = NULL;


   //-----
   // Allocate memory resources
   //-----
	
   h_cpuElems  = (U32 *)malloc( mem_size_elems );
   h_gpuCounts = (U32 *)malloc( mem_size_counts );
   h_cpuCounts = (U32 *)malloc( mem_size_counts );


   //-----
   // Transfer arrays from GPU to CPU
   //-----

	cutilSafeCall( cudaMemcpy( h_cpuElems,  d_gpuElems,  mem_size_elems, cudaMemcpyDeviceToHost) );
	cutilSafeCall( cudaMemcpy( h_gpuCounts, d_gpuCounts, mem_size_counts, cudaMemcpyDeviceToHost) );

   // Zero CPU counts
	for (U32 idx = 0; idx < 256u; idx++)
	{
		h_cpuCounts[idx] = 0u;
	}


   // Get TRISH types
   typedef ExtractorBytes<U32> Extractor;

   typedef typename TRISH_trait<valT>:base_type     baseType;
   typedef typename TRISH_trait<valT>::bin_type     binType;
   typedef typename TRISH_trait<valT>::upscale_type upscaleType;
   typedef typename TRISH_trait<valT>::convert_type convertType;


   //-----
	// Compute CPU row counts
   //-----


   // Initialize Mapper
   mapT mapper;
   mapper.Initiate( minVal, maxVal, numBins );

   U32 val1;
   upscaleType b1, b2, b3, b4;
   binType bin1, bin2, bin3, bin4;
	for (U32 idx = 0; idx < nElems; idx+=1u)
	{
      // Get current values
		val1 = h_cpuElems[idx];

      // Extract 4 bytes from single values
      Extractor.Extract4( b1, b2, b3, b4, val2 );

      // Transform values into bins
      mapper.Transform4( bin1, bin2, bin3, bin4,   // Out => transformed bins
                         b1, b2, b3, b4 );         // In  => values to transform into bins

      // Bin results
		h_cpuCounts[bin1] += 1u;
      h_cpuCounts[bin2] += 1u;
      h_cpuCounts[bin3] += 1u;
      h_cpuCounts[bin4] += 1u;
	}

   // Cleanup Mapper
   mapper.Finish();


   //-----
	// Compare CPU vs. GPU totals
   //-----

   U64 totalCPU = 0ull;
   U64 totalGPU = 0ull;
	for (U32 idx = 0; idx < numBins; idx++)
	{
		U32 cpuCount = h_cpuCounts[idx];
		U32 gpuCount = h_gpuCounts[idx];

      totalCPU += (U64)cpuCount;
      totalGPU += (U64)gpuCount;

		if (cpuCount != gpuCount)
		{
			fprintf( stdout, "Total Counts[%u] : CPU (%u) != GPU (%u) !!! ERROR !!!\n",
				      idx, cpuCount, gpuCount );
		}
		else
		{
			fprintf( stdout, "Total Counts[%u] : CPU (%u) == GPU (%u) Success\n",
				      idx, cpuCount, gpuCount );
		}
	}

   // Get items below range
   U32 minCPU, minGPU;
   minCPU = h_cpuCounts[numBins+1];
   minGPU = h_gpuCounts[numBins+1];
   if (minCPU != minGPU)
   {
      fprintf( stdout, "For < min (%d), CPU count (%d) != GPU count (%d)  !!! ERROR !!!\n",
               minVal, minCPU, minGPU );
   }
   else
   {
      fprintf( stdout, "For < min (%d), CPU count (%d) == GPU count (%d)  Success\n",
               minVal, minCPU, minGPU );
   }
   totalCPU += (U64)minCPU;
   totalGPU += (U64)minGPU;


   // Get items above range
   U32 maxCPU, maxGPU;
   maxCPU = h_cpuCounts[numBins+2];
   maxGPU = h_gpuCounts[numBins+2];
   if (minCPU != minGPU)
   {
      fprintf( stdout, "For > max (%d), CPU count (%d) != GPU count (%d)  !!! ERROR !!!\n",
               maxVal, maxCPU, maxGPU );
   }
   else
   {
      fprintf( stdout, "For > max (%d), CPU count (%d) == GPU count (%d)  Success\n",
               maxVal, maxCPU, maxGPU );
   }
   totalCPU += (U64)maxCPU;
   totalGPU += (U64)maxGPU;

   // Verify final counts
   if (totalCPU != totalGPU)
   {
      fprintf( stdout, "\nTotal CPU (%I64u) != Total GPU (%I64u)  !!! ERROR !!!\n\n\n",
               totalCPU, totalGPU );
   }
   else
   {
      fprintf( stdout, "\nTotal CPU (%I64u) == Total GPU (%I64u)  Success\n\n\n",
               totalCPU, totalGPU );
   }

   //-----
   // Free memory resources
   //-----

   free( h_cpuCounts );
	free( h_gpuCounts );
	free( h_cpuElems  );
}


// Verify 2 byte integers (I16, U16)
template < 
           typename valT,     // Underlying value type
           typename mapT      // Mapper Type
         >
__host__
void TRISH_VerifyHistogram_B2
( 
	 U32   nElems,          // IN - number of 32-bit elements to bin & count
	 U32 * d_gpuElems,      // IN - array of elements to bin & count
    U32   numBins,         // IN - number of bins in histogram
	 U32 * d_gpuCounts,     // IN - GPU histogram counts
    valT  minVal,          // IN - [min,max] values for histogram
    valT  maxVal           //      ditto
)
{
   assert( numBins > 0u );
   assert( numBins <= 256u );
   assert( nElems > 0u );

   U32 mem_size_elems  = nElems * sizeof( U32 );
   U32 mem_size_counts = 256u * sizeof( U32 );

   U32 * h_cpuElems  = NULL;
   U32 * h_gpuCounts = NULL;
   U32 * h_cpuCounts = NULL;


   //-----
   // Allocate memory resources
   //-----
	
   h_cpuElems  = (U32 *)malloc( mem_size_elems );
   h_gpuCounts = (U32 *)malloc( mem_size_counts );
   h_cpuCounts = (U32 *)malloc( mem_size_counts );


   //-----
   // Transfer arrays from GPU to CPU
   //-----

	cutilSafeCall( cudaMemcpy( h_cpuElems,  d_gpuElems,  mem_size_elems, cudaMemcpyDeviceToHost) );
	cutilSafeCall( cudaMemcpy( h_gpuCounts, d_gpuCounts, mem_size_counts, cudaMemcpyDeviceToHost) );

   // Zero CPU counts
	for (U32 idx = 0; idx < 256u; idx++)
	{
		h_cpuCounts[idx] = 0u;
	}


   // Get TRISH types
   typedef ExtractorWords<U32> Extractor;

   typedef typename TRISH_trait<valT>:base_type     baseType;
   typedef typename TRISH_trait<valT>::bin_type     binType;
   typedef typename TRISH_trait<valT>::upscale_type upscaleType;
   typedef typename TRISH_trait<valT>::convert_type convertType;


   //-----
	// Compute CPU row counts
   //-----


   // Initialize Mapper
   mapT mapper;
   mapper.Initiate( minVal, maxVal, numBins );

   U32 val1, val2;
   upscaleType b1, b2, b3, b4;
   binType bin1, bin2, bin3, bin4;
	for (U32 idx = 0u; idx < nElems; idx+=2u)
	{
      // Get current value
		val1 = h_cpuElems[idx];
      val2 = h_cpuElems[idx+1u];

      // Extract 4 words from 2 values
      Extractor::Extract4( b1, b2, b3, b4, val1, val2 );

      // Transform 
      mapper.Transform4( bin1, bin2, bin3, bin4,   // Out => transformed bins
                         b1, b2, b3, b4 );         // In  => values to transform into bins

      // Bin results
		h_cpuCounts[bin1] += 1u;
      h_cpuCounts[bin2] += 1u;
      h_cpuCounts[bin3] += 1u;
      h_cpuCounts[bin4] += 1u;
	}

   // Cleanup Mapper
   mapper.Finish();


   //-----
	// Compare CPU vs. GPU totals
   //-----

   U64 totalCPU = 0ull;
   U64 totalGPU = 0ull;
	for (U32 idx = 0; idx < numBins; idx++)
	{
		U32 cpuCount = h_cpuCounts[idx];
		U32 gpuCount = h_gpuCounts[idx];

      totalCPU += (U64)cpuCount;
      totalGPU += (U64)gpuCount;

		if (cpuCount != gpuCount)
		{
			fprintf( stdout, "Total Counts[%u] : CPU (%u) != GPU (%u) !!! ERROR !!!\n",
				      idx, cpuCount, gpuCount );
		}
		else
		{
			fprintf( stdout, "Total Counts[%u] : CPU (%u) == GPU (%u) Success\n",
				      idx, cpuCount, gpuCount );
		}
	}

   // Get items below range
   U32 minCPU, minGPU;
   minCPU = h_cpuCounts[numBins+1];
   minGPU = h_gpuCounts[numBins+1];
   if (minCPU != minGPU)
   {
      fprintf( stdout, "For < min (%d), CPU count (%d) != GPU count (%d)  !!! ERROR !!!\n",
               minVal, minCPU, minGPU );
   }
   else
   {
      fprintf( stdout, "For < min (%d), CPU count (%d) == GPU count (%d)  Success\n",
               minVal, minCPU, minGPU );
   }
   totalCPU += (U64)minCPU;
   totalGPU += (U64)minGPU;


   // Get items above range
   U32 maxCPU, maxGPU;
   maxCPU = h_cpuCounts[numBins+2];
   maxGPU = h_gpuCounts[numBins+2];
   if (minCPU != minGPU)
   {
      fprintf( stdout, "For > max (%d), CPU count (%d) != GPU count (%d)  !!! ERROR !!!\n",
               maxVal, maxCPU, maxGPU );
   }
   else
   {
      fprintf( stdout, "For > max (%d), CPU count (%d) == GPU count (%d)  Success\n",
               maxVal, maxCPU, maxGPU );
   }
   totalCPU += (U64)maxCPU;
   totalGPU += (U64)maxGPU;

   // Verify final counts
   if (totalCPU != totalGPU)
   {
      fprintf( stdout, "\nTotal CPU (%I64u) != Total GPU (%I64u)  !!! ERROR !!!\n\n\n",
               totalCPU, totalGPU );
   }
   else
   {
      fprintf( stdout, "\nTotal CPU (%I64u) == Total GPU (%I64u)  Success\n\n\n",
               totalCPU, totalGPU );
   }

   //-----
   // Free memory resources
   //-----

   free( h_cpuCounts );
	free( h_gpuCounts );
	free( h_cpuElems  );
}


// Verify 4 byte integers (I32, U32)
template < 
           typename valT,     // Underlying value type
           typename mapT      // Mapper Type
         >
__host__
void TRISH_VerifyHistogram_B4
( 
	 U32    nElems,          // IN - number of 32-bit elements to bin & count
	 valT * d_gpuElems,      // IN - array of elements to bin & count
    U32    numBins,         // IN - number of bins in histogram
	 U32 *  d_gpuCounts,     // IN - GPU histogram counts
    valT   minVal,          // IN - [min,max] values for histogram
    valT   maxVal           //      ditto
)
{
   assert( numBins > 0u );
   assert( numBins <= 256u );
   assert( nElems > 0u );

   U32 mem_size_elems  = nElems * sizeof( valT );
   U32 mem_size_counts = 256u * sizeof( U32 );

   valT * h_cpuElems  = NULL;
   U32  * h_gpuCounts = NULL;
   U32  * h_cpuCounts = NULL;


   //-----
   // Allocate memory resources
   //-----
	
   h_cpuElems  = (U32 *)malloc( mem_size_elems );
   h_gpuCounts = (U32 *)malloc( mem_size_counts );
   h_cpuCounts = (U32 *)malloc( mem_size_counts );


   //-----
   // Transfer arrays from GPU to CPU
   //-----

	cutilSafeCall( cudaMemcpy( h_cpuElems,  d_gpuElems,  mem_size_elems, cudaMemcpyDeviceToHost) );
	cutilSafeCall( cudaMemcpy( h_gpuCounts, d_gpuCounts, mem_size_counts, cudaMemcpyDeviceToHost) );

   // Zero CPU counts
	for (U32 idx = 0; idx < 256u; idx++)
	{
		h_cpuCounts[idx] = 0u;
	}


   // Get TRISH types
   typedef ExtractorWords<valT> Extractor;

   typedef typename TRISH_trait<valT>:base_type     baseType;
   typedef typename TRISH_trait<valT>::bin_type     binType;
   typedef typename TRISH_trait<valT>::upscale_type upscaleType;
   typedef typename TRISH_trait<valT>::convert_type convertType;


   //-----
	// Compute CPU row counts
   //-----


   // Initialize Mapper
   mapT mapper;
   mapper.Initiate( minVal, maxVal, numBins );

   U32 nRows = nElems / 4u;
   U32 nCols = nElems % 4u;
   valT val1, val2, val3, val4;
   upscaleType b1, b2, b3, b4;
   binType bin1, bin2, bin3, bin4;
	for (U32 idx = 0u; idx < nRows; idx+=4u)
	{
      // Get current value
		val1 = h_cpuElems[idx];
      val2 = h_cpuElems[idx+1u];
      val3 = h_cpuElems[idx+2u];
      val4 = h_cpuElems[idx+3u];

      b1 = (upscaleType)val1;
      b2 = (upscaleType)val2;
      b3 = (upscaleType)val3;
      b4 = (upscaleType)val4;

      // Transform 
      mapper.Transform4( bin1, bin2, bin3, bin4,   // Out => transformed bins
                         b1, b2, b3, b4 );         // In  => values to transform into bins

      // Bin results
		h_cpuCounts[bin1] += 1u;
      h_cpuCounts[bin2] += 1u;
      h_cpuCounts[bin3] += 1u;
      h_cpuCounts[bin4] += 1u;
	}

   // Cleanup Mapper
   mapper.Finish();


   //-----
	// Compare CPU vs. GPU totals
   //-----

   U64 totalCPU = 0ull;
   U64 totalGPU = 0ull;
	for (U32 idx = 0; idx < numBins; idx++)
	{
		U32 cpuCount = h_cpuCounts[idx];
		U32 gpuCount = h_gpuCounts[idx];

      totalCPU += (U64)cpuCount;
      totalGPU += (U64)gpuCount;

		if (cpuCount != gpuCount)
		{
			fprintf( stdout, "Total Counts[%u] : CPU (%u) != GPU (%u) !!! ERROR !!!\n",
				      idx, cpuCount, gpuCount );
		}
		else
		{
			fprintf( stdout, "Total Counts[%u] : CPU (%u) == GPU (%u) Success\n",
				      idx, cpuCount, gpuCount );
		}
	}

   // Get items below range
   U32 minCPU, minGPU;
   minCPU = h_cpuCounts[numBins+1];
   minGPU = h_gpuCounts[numBins+1];
   if (minCPU != minGPU)
   {
      fprintf( stdout, "For < min (%d), CPU count (%d) != GPU count (%d)  !!! ERROR !!!\n",
               minVal, minCPU, minGPU );
   }
   else
   {
      fprintf( stdout, "For < min (%d), CPU count (%d) == GPU count (%d)  Success\n",
               minVal, minCPU, minGPU );
   }
   totalCPU += (U64)minCPU;
   totalGPU += (U64)minGPU;


   // Get items above range
   U32 maxCPU, maxGPU;
   maxCPU = h_cpuCounts[numBins+2];
   maxGPU = h_gpuCounts[numBins+2];
   if (minCPU != minGPU)
   {
      fprintf( stdout, "For > max (%d), CPU count (%d) != GPU count (%d)  !!! ERROR !!!\n",
               maxVal, maxCPU, maxGPU );
   }
   else
   {
      fprintf( stdout, "For > max (%d), CPU count (%d) == GPU count (%d)  Success\n",
               maxVal, maxCPU, maxGPU );
   }
   totalCPU += (U64)maxCPU;
   totalGPU += (U64)maxGPU;

   // Verify final counts
   if (totalCPU != totalGPU)
   {
      fprintf( stdout, "\nTotal CPU (%I64u) != Total GPU (%I64u)  !!! ERROR !!!\n\n\n",
               totalCPU, totalGPU );
   }
   else
   {
      fprintf( stdout, "\nTotal CPU (%I64u) == Total GPU (%I64u)  Success\n\n\n",
               totalCPU, totalGPU );
   }

   //-----
   // Free memory resources
   //-----

   free( h_cpuCounts );
	free( h_gpuCounts );
	free( h_cpuElems  );
}
#endif


/*-----------------------------------------------
  Name:  BinCounts
  Desc:  Adds Bins into count array
-----------------------------------------------*/

template <U32 BlockSize>
__host__ __device__ __forceinline__
void BinCount1 
( 
	U32 * cntPtr, // OUT - count array (to store bin results in)
	U32   bin1    // IN  - input 'bins' to count
) 
{
	// Lane Row[0..63] = bin / 4
	U32 LI_1;
	LI_1 = bin1 >> 2u;

	// Multiply by block Size;
	LI_1 = LI_1 * BlockSize;

	// Lane Col[0,1,2,3] = bin % 4
   U32 col1;
	col1 = bin1 & 0x3u;

	// Shift[0,8,16,24] = Lane Col [0,1,2,3] * 8
	U32 s1;
	s1 = col1 << 3u;

	// Get Increments
	U32 inc1;
	inc1 = 1u << s1;
   
	U32 oldCnt, newCnt;

	//-----
	// Add bin counts into count array
	//-----

	// Increment 1st bin count
	oldCnt = cntPtr[LI_1];
	newCnt = oldCnt + inc1;
	cntPtr[LI_1] = newCnt;
}


template <U32 BlockSize>
__host__ __device__ __forceinline__
void BinCount2 
( 
	U32 * cntPtr, // OUT - count array (to store bin results in)
	U32   bin1,   // IN  - input 'bins' to count
	U32   bin2   
) 
{
	// Lane Row = bin / 4
	U32 LI_1, LI_2;
	LI_1 = bin1 >> 2u;
	LI_2 = bin2 >> 2u;

	// Multiply by block Size;
	LI_1 = LI_1 * BlockSize;
	LI_2 = LI_2 * BlockSize;

	// Lane Col = bin % 4
   U32 col1, col2;
	col1 = bin1 & 0x3u;
	col2 = bin2 & 0x3u;

	// Shift = Lane Col [0,1,2,3] * 8
	U32 s1, s2;
	s1 = col1 << 3u;
	s2 = col2 << 3u;

	// Get Increments
	U32 inc1, inc2;
	inc1 = 1u << s1;
	inc2 = 1u << s2;


	//-----
	// Add bin counts into count array
	//-----

	U32 oldCnt, newCnt;

	// Increment 1st bin
	oldCnt = cntPtr[LI_1];
	newCnt = oldCnt + inc1;
	cntPtr[LI_1] = newCnt;

	// Increment 2nd bin
	oldCnt = cntPtr[LI_2];
	newCnt = oldCnt + inc2;
	cntPtr[LI_2] = newCnt;
}


template <U32 BlockSize>
__host__ __device__ __forceinline__
void BinCount3 
( 
	U32 * cntPtr, // OUT - count array (to store bin results in)
	U32   bin1,   // IN  - input 'bins' to count
	U32   bin2,   
	U32   bin3
) 
{
	// Lane Row = bin / 4
	U32 LI_1, LI_2, LI_3;
	LI_1 = bin1 >> 2u;
	LI_2 = bin2 >> 2u;
	LI_3 = bin3 >> 2u;

	// Multiply by block Size;
	LI_1 = LI_1 * BlockSize;
	LI_2 = LI_2 * BlockSize;
	LI_3 = LI_3 * BlockSize;

	// Lane Col = bin % 4
   U32 col1, col2, col3;
	col1 = bin1 & 0x3u;
	col2 = bin2 & 0x3u;
	col3 = bin3 & 0x3u;

	// Shift = Lane Col [0,1,2,3] * 8
	U32 s1, s2, s3;
	s1 = col1 << 3u;
	s2 = col2 << 3u;
	s3 = col3 << 3u;

	// Get Increments
	U32 inc1, inc2, inc3;
	inc1 = 1u << s1;
	inc2 = 1u << s2;
	inc3 = 1u << s3;


	//-----
	// Add bin counts into count array
	//-----

	U32 oldCnt, newCnt;

	// Increment 1st bin
	oldCnt = cntPtr[LI_1];
	newCnt = oldCnt + inc1;
	cntPtr[LI_1] = newCnt;

	// Increment 2nd bin
	oldCnt = cntPtr[LI_2];
	newCnt = oldCnt + inc2;
	cntPtr[LI_2] = newCnt;

	// Increment 3rd bin
	oldCnt = cntPtr[LI_3];
	newCnt = oldCnt + inc3;
	cntPtr[LI_3] = newCnt;
}


template <U32 BlockSize>
__host__ __device__ __forceinline__
void BinCount4 
( 
	U32 * cntPtr, // OUT - count array (to store bin results in)
	U32   bin1,   // IN  - input 'bins' to count
	U32   bin2,   
	U32   bin3,
	U32   bin4
) 
{
	// Lane Row = bin / 4
	U32 LI_1, LI_2, LI_3, LI_4;
	LI_1 = bin1 >> 2u;
	LI_2 = bin2 >> 2u;
	LI_3 = bin3 >> 2u;
	LI_4 = bin4 >> 2u;

	// Multiply by block Size;
	LI_1 = LI_1 * BlockSize;
	LI_2 = LI_2 * BlockSize;
	LI_3 = LI_3 * BlockSize;
	LI_4 = LI_4 * BlockSize;

	// Lane Col = bin % 4
   U32 col1, col2, col3, col4;
	col1 = bin1 & 0x3u;
	col2 = bin2 & 0x3u;
	col3 = bin3 & 0x3u;
	col4 = bin4 & 0x3u;

	// Shift = Lane Col [0,1,2,3] * 8
	U32 s1, s2, s3, s4;
	s1 = col1 << 3u;
	s2 = col2 << 3u;
	s3 = col3 << 3u;
	s4 = col4 << 3u;

	// Get Increments
	U32 inc1, inc2, inc3, inc4;
	inc1 = 1u << s1;
	inc2 = 1u << s2;
	inc3 = 1u << s3;
	inc4 = 1u << s4;


	//-----
	// Add bin counts into count array
	//-----

	U32 oldCnt, newCnt;

	// Increment 1st bin
	oldCnt = cntPtr[LI_1];
	newCnt = oldCnt + inc1;
	cntPtr[LI_1] = newCnt;

	// Increment 2nd bin
	oldCnt = cntPtr[LI_2];
	newCnt = oldCnt + inc2;
	cntPtr[LI_2] = newCnt;

	// Increment 3rd bin
	oldCnt = cntPtr[LI_3];
	newCnt = oldCnt + inc3;
	cntPtr[LI_3] = newCnt;

	// Increment 4th bin
	oldCnt = cntPtr[LI_4];
	newCnt = oldCnt + inc4;
	cntPtr[LI_4] = newCnt;
}


/*---------------------------------------------------------
  Name:   SetArray_BlockSeq
  Desc:   Sets elements in array to specified value
  Note:   Uses "Block Sequential" access pattern
 ---------------------------------------------------------*/

template <  
            typename valT,		// Underlying value type
            U32 BlockSize,    // Threads Per Block
            U32 nSafePasses,  // Number of safe passes
            U32 nLeftOver,    // Number of left over elements
            U32 maxSize       // Max Size of array
         >
__device__ __forceinline__
void SetArray_BlockSeq
( 
   valT * basePtr,      // IN/OUT - array to set to 'set' value
   valT   toSet         // IN - value to set array elements 'to'
) 
{
   // Get 'per thread' pointer
   valT * setPtr = basePtr + threadIdx.x;

		// Initialize as many elements as we
		// safely can with no range checking
	if (nSafePasses >=  1u) { setPtr[( 0u * BlockSize)] = toSet; }
	if (nSafePasses >=  2u) { setPtr[( 1u * BlockSize)] = toSet; }
	if (nSafePasses >=  3u) { setPtr[( 2u * BlockSize)] = toSet; }
	if (nSafePasses >=  4u) { setPtr[( 3u * BlockSize)] = toSet; }
	if (nSafePasses >=  5u) { setPtr[( 4u * BlockSize)] = toSet; }
	if (nSafePasses >=  6u) { setPtr[( 5u * BlockSize)] = toSet; }
	if (nSafePasses >=  7u) { setPtr[( 6u * BlockSize)] = toSet; }
	if (nSafePasses >=  8u) { setPtr[( 7u * BlockSize)] = toSet; }
	if (nSafePasses >=  9u) { setPtr[( 8u * BlockSize)] = toSet; }
	if (nSafePasses >= 10u) { setPtr[( 9u * BlockSize)] = toSet; }
	if (nSafePasses >= 11u) { setPtr[(10u * BlockSize)] = toSet; }
	if (nSafePasses >= 12u) { setPtr[(11u * BlockSize)] = toSet; }
	if (nSafePasses >= 13u) { setPtr[(12u * BlockSize)] = toSet; }
	if (nSafePasses >= 14u) { setPtr[(13u * BlockSize)] = toSet; }
	if (nSafePasses >= 15u) { setPtr[(14u * BlockSize)] = toSet; }
	if (nSafePasses >= 16u) { setPtr[(15u * BlockSize)] = toSet; }
	if (nSafePasses >= 17u) { setPtr[(16u * BlockSize)] = toSet; }
	if (nSafePasses >= 18u) { setPtr[(17u * BlockSize)] = toSet; }
	if (nSafePasses >= 19u) { setPtr[(18u * BlockSize)] = toSet; }
	if (nSafePasses >= 20u) { setPtr[(19u * BlockSize)] = toSet; }
	if (nSafePasses >= 21u) { setPtr[(20u * BlockSize)] = toSet; }
	if (nSafePasses >= 22u) { setPtr[(21u * BlockSize)] = toSet; }
	if (nSafePasses >= 23u) { setPtr[(22u * BlockSize)] = toSet; }
	if (nSafePasses >= 24u) { setPtr[(23u * BlockSize)] = toSet; }
	if (nSafePasses >= 25u) { setPtr[(24u * BlockSize)] = toSet; }
	if (nSafePasses >= 26u) { setPtr[(25u * BlockSize)] = toSet; }
	if (nSafePasses >= 27u) { setPtr[(26u * BlockSize)] = toSet; }
	if (nSafePasses >= 28u) { setPtr[(27u * BlockSize)] = toSet; }
	if (nSafePasses >= 29u) { setPtr[(28u * BlockSize)] = toSet; }
	if (nSafePasses >= 30u) { setPtr[(29u * BlockSize)] = toSet; }
	if (nSafePasses >= 31u) { setPtr[(30u * BlockSize)] = toSet; }
	if (nSafePasses >= 32u) { setPtr[(31u * BlockSize)] = toSet; }
	if (nSafePasses >= 33u) { setPtr[(32u * BlockSize)] = toSet; }
	if (nSafePasses >= 34u) { setPtr[(33u * BlockSize)] = toSet; }
	if (nSafePasses >= 35u) { setPtr[(34u * BlockSize)] = toSet; }
	if (nSafePasses >= 36u) { setPtr[(35u * BlockSize)] = toSet; }
	if (nSafePasses >= 37u) { setPtr[(36u * BlockSize)] = toSet; }
	if (nSafePasses >= 38u) { setPtr[(37u * BlockSize)] = toSet; }
	if (nSafePasses >= 39u) { setPtr[(38u * BlockSize)] = toSet; }
	if (nSafePasses >= 40u) { setPtr[(39u * BlockSize)] = toSet; }
	if (nSafePasses >= 41u) { setPtr[(40u * BlockSize)] = toSet; }
	if (nSafePasses >= 42u) { setPtr[(41u * BlockSize)] = toSet; }
	if (nSafePasses >= 43u) { setPtr[(42u * BlockSize)] = toSet; }
	if (nSafePasses >= 44u) { setPtr[(43u * BlockSize)] = toSet; }
	if (nSafePasses >= 45u) { setPtr[(44u * BlockSize)] = toSet; }
	if (nSafePasses >= 46u) { setPtr[(45u * BlockSize)] = toSet; }
	if (nSafePasses >= 47u) { setPtr[(46u * BlockSize)] = toSet; }
	if (nSafePasses >= 48u) { setPtr[(47u * BlockSize)] = toSet; }
	if (nSafePasses >= 49u) { setPtr[(48u * BlockSize)] = toSet; }
	if (nSafePasses >= 50u) { setPtr[(49u * BlockSize)] = toSet; }
	if (nSafePasses >= 51u) { setPtr[(50u * BlockSize)] = toSet; }
	if (nSafePasses >= 52u) { setPtr[(51u * BlockSize)] = toSet; }
	if (nSafePasses >= 53u) { setPtr[(52u * BlockSize)] = toSet; }
	if (nSafePasses >= 54u) { setPtr[(53u * BlockSize)] = toSet; }
	if (nSafePasses >= 55u) { setPtr[(54u * BlockSize)] = toSet; }
	if (nSafePasses >= 56u) { setPtr[(55u * BlockSize)] = toSet; }
	if (nSafePasses >= 57u) { setPtr[(56u * BlockSize)] = toSet; }
	if (nSafePasses >= 58u) { setPtr[(57u * BlockSize)] = toSet; }
	if (nSafePasses >= 59u) { setPtr[(58u * BlockSize)] = toSet; }
	if (nSafePasses >= 60u) { setPtr[(59u * BlockSize)] = toSet; }
	if (nSafePasses >= 61u) { setPtr[(60u * BlockSize)] = toSet; }
	if (nSafePasses >= 62u) { setPtr[(61u * BlockSize)] = toSet; }
	if (nSafePasses >= 63u) { setPtr[(62u * BlockSize)] = toSet; }
	if (nSafePasses >= 64u) { setPtr[(63u * BlockSize)] = toSet; }
	if (nSafePasses >= 65u) { setPtr[(64u * BlockSize)] = toSet; }
	if (nSafePasses >= 66u) { setPtr[(65u * BlockSize)] = toSet; }

	// Set any 'left over' values with range checking
	if (nLeftOver > 0u)
	{ 
		U32 idx = (nSafePasses * BlockSize) + threadIdx.x;
		if (idx < maxSize)
		{
			basePtr[idx] = toSet;
		}
	}
}


/*---------------------------------------------------------
  Name:   SetArray_WarpSeq
  Desc:   Sets elements in array to specified value
  Note:   Uses "Warp Sequential" access pattern
 ---------------------------------------------------------*/

template <  
           typename valT,	  // Underlying value type
           U32 WarpSize,     // Threads per Warp
           U32 nSafePasses,  // Number of safe passes (warps per subsection)
           U32 nLeftOver,    // Number of left over elements
           U32 maxSize       // Max Size of array
         >
__device__ __forceinline__
void SetArray_WarpSeq
( 
   valT * basePtr,      // IN/OUT - array to set to 'set' value
   valT   toSet,        // IN - value to set array elements 'to'
   U32    startIdx      // starting index for this thread
) 
{
   // Get 'per thread' pointer
   valT * setPtr  = &basePtr[startIdx];

		// Initialize as many elements as we
		// safely can with no range checking
	if (nSafePasses >=  1u) { setPtr[( 0u * WarpSize)] = toSet; }
	if (nSafePasses >=  2u) { setPtr[( 1u * WarpSize)] = toSet; }
	if (nSafePasses >=  3u) { setPtr[( 2u * WarpSize)] = toSet; }
	if (nSafePasses >=  4u) { setPtr[( 3u * WarpSize)] = toSet; }
	if (nSafePasses >=  5u) { setPtr[( 4u * WarpSize)] = toSet; }
	if (nSafePasses >=  6u) { setPtr[( 5u * WarpSize)] = toSet; }
	if (nSafePasses >=  7u) { setPtr[( 6u * WarpSize)] = toSet; }
	if (nSafePasses >=  8u) { setPtr[( 7u * WarpSize)] = toSet; }
	if (nSafePasses >=  9u) { setPtr[( 8u * WarpSize)] = toSet; }
	if (nSafePasses >= 10u) { setPtr[( 9u * WarpSize)] = toSet; }
	if (nSafePasses >= 11u) { setPtr[(10u * WarpSize)] = toSet; }
	if (nSafePasses >= 12u) { setPtr[(11u * WarpSize)] = toSet; }
	if (nSafePasses >= 13u) { setPtr[(12u * WarpSize)] = toSet; }
	if (nSafePasses >= 14u) { setPtr[(13u * WarpSize)] = toSet; }
	if (nSafePasses >= 15u) { setPtr[(14u * WarpSize)] = toSet; }
	if (nSafePasses >= 16u) { setPtr[(15u * WarpSize)] = toSet; }
	if (nSafePasses >= 17u) { setPtr[(16u * WarpSize)] = toSet; }
	if (nSafePasses >= 18u) { setPtr[(17u * WarpSize)] = toSet; }
	if (nSafePasses >= 19u) { setPtr[(18u * WarpSize)] = toSet; }
	if (nSafePasses >= 20u) { setPtr[(19u * WarpSize)] = toSet; }
	if (nSafePasses >= 21u) { setPtr[(20u * WarpSize)] = toSet; }
	if (nSafePasses >= 22u) { setPtr[(21u * WarpSize)] = toSet; }
	if (nSafePasses >= 23u) { setPtr[(22u * WarpSize)] = toSet; }
	if (nSafePasses >= 24u) { setPtr[(23u * WarpSize)] = toSet; }
	if (nSafePasses >= 25u) { setPtr[(24u * WarpSize)] = toSet; }
	if (nSafePasses >= 26u) { setPtr[(25u * WarpSize)] = toSet; }
	if (nSafePasses >= 27u) { setPtr[(26u * WarpSize)] = toSet; }
	if (nSafePasses >= 28u) { setPtr[(27u * WarpSize)] = toSet; }
	if (nSafePasses >= 29u) { setPtr[(28u * WarpSize)] = toSet; }
	if (nSafePasses >= 30u) { setPtr[(29u * WarpSize)] = toSet; }
	if (nSafePasses >= 31u) { setPtr[(30u * WarpSize)] = toSet; }
	if (nSafePasses >= 32u) { setPtr[(31u * WarpSize)] = toSet; }
	if (nSafePasses >= 33u) { setPtr[(32u * WarpSize)] = toSet; }
	if (nSafePasses >= 34u) { setPtr[(33u * WarpSize)] = toSet; }
	if (nSafePasses >= 35u) { setPtr[(34u * WarpSize)] = toSet; }
	if (nSafePasses >= 36u) { setPtr[(35u * WarpSize)] = toSet; }
	if (nSafePasses >= 37u) { setPtr[(36u * WarpSize)] = toSet; }
	if (nSafePasses >= 38u) { setPtr[(37u * WarpSize)] = toSet; }
	if (nSafePasses >= 39u) { setPtr[(38u * WarpSize)] = toSet; }
	if (nSafePasses >= 40u) { setPtr[(39u * WarpSize)] = toSet; }
	if (nSafePasses >= 41u) { setPtr[(40u * WarpSize)] = toSet; }
	if (nSafePasses >= 42u) { setPtr[(41u * WarpSize)] = toSet; }
	if (nSafePasses >= 43u) { setPtr[(42u * WarpSize)] = toSet; }
	if (nSafePasses >= 44u) { setPtr[(43u * WarpSize)] = toSet; }
	if (nSafePasses >= 45u) { setPtr[(44u * WarpSize)] = toSet; }
	if (nSafePasses >= 46u) { setPtr[(45u * WarpSize)] = toSet; }
	if (nSafePasses >= 47u) { setPtr[(46u * WarpSize)] = toSet; }
	if (nSafePasses >= 48u) { setPtr[(47u * WarpSize)] = toSet; }
	if (nSafePasses >= 49u) { setPtr[(48u * WarpSize)] = toSet; }
	if (nSafePasses >= 50u) { setPtr[(49u * WarpSize)] = toSet; }
	if (nSafePasses >= 51u) { setPtr[(50u * WarpSize)] = toSet; }
	if (nSafePasses >= 52u) { setPtr[(51u * WarpSize)] = toSet; }
	if (nSafePasses >= 53u) { setPtr[(52u * WarpSize)] = toSet; }
	if (nSafePasses >= 54u) { setPtr[(53u * WarpSize)] = toSet; }
	if (nSafePasses >= 55u) { setPtr[(54u * WarpSize)] = toSet; }
	if (nSafePasses >= 56u) { setPtr[(55u * WarpSize)] = toSet; }
	if (nSafePasses >= 57u) { setPtr[(56u * WarpSize)] = toSet; }
	if (nSafePasses >= 58u) { setPtr[(57u * WarpSize)] = toSet; }
	if (nSafePasses >= 59u) { setPtr[(58u * WarpSize)] = toSet; }
	if (nSafePasses >= 60u) { setPtr[(59u * WarpSize)] = toSet; }
	if (nSafePasses >= 61u) { setPtr[(60u * WarpSize)] = toSet; }
	if (nSafePasses >= 62u) { setPtr[(61u * WarpSize)] = toSet; }
	if (nSafePasses >= 63u) { setPtr[(62u * WarpSize)] = toSet; }
	if (nSafePasses >= 64u) { setPtr[(63u * WarpSize)] = toSet; }
	if (nSafePasses >= 65u) { setPtr[(64u * WarpSize)] = toSet; }
	if (nSafePasses >= 66u) { setPtr[(65u * WarpSize)] = toSet; }

	// Set any 'left over' values with range checking
	if (nLeftOver > 0u)
	{ 
		U32 idx = startIdx + (nSafePasses * WarpSize);
		if (idx < maxSize)
		{
			basePtr[idx] = toSet;
		}
	}
}


/*-------------------------------------------------------------------
  Name:   SS_Sums_4_Next_V1
  Desc:   Serial scan on next 4 elements in seq [0..3]
 ------------------------------------------------------------------*/

template < 
           U32 BlockSize,     // Threads per block
           U32 BlockMask      // Block Mask
         >
__device__ __forceinline__
void SS_Sums_4_Next_V1
( 
   U32 & sum1,     // OUT - sum1 .. sum4 (as singletons)
   U32 & sum2,
   U32 & sum3,
   U32 & sum4,
   U32 * cntPtr,   // IN  - 'per thread' counts <horizontal row> to sum up
   U32   baseIdx
) 
{
   // wrap = (idx + [0..3]) % BlockSize
   U32 idx1, idx2, idx3, idx4;
   idx1 = baseIdx + 0u;
   idx2 = baseIdx + 1u;
   idx3 = baseIdx + 2u;
   idx4 = baseIdx + 3u;

   U32 wrap1, wrap2, wrap3, wrap4;
   wrap1 = idx1 & BlockMask;
   wrap2 = idx2 & BlockMask;
   wrap3 = idx3 & BlockMask;
   wrap4 = idx4 & BlockMask;

   //-
   // Grab 4 elements in seq [0..3]
   //-

   U32 lane1, lane2, lane3, lane4;
   lane1 = cntPtr[wrap1];
   lane2 = cntPtr[wrap2];
   lane3 = cntPtr[wrap3];
   lane4 = cntPtr[wrap4];


   //-
   // Zero out sequence [0..3]
   //-

   cntPtr[wrap1] = 0u;
   cntPtr[wrap2] = 0u;
   cntPtr[wrap3] = 0u;
   cntPtr[wrap4] = 0u;


   //-
   // Accumulate all 4 groups in each lane
   //-

   //-
   // Initialize sums from 1st lane (of 4 groups)
   //-
   U32 s3 = lane1 >> 16u;     // 3rd bin (of 4) in lane
   U32 s2 = lane1 >>  8u;     // 2nd bin (of 4) in lane

   U32 cnt4 = lane1 >> 24u;
   U32 cnt3 = s3 & 0xFFu;
   U32 cnt2 = s2 & 0xFFu;
   U32 cnt1 = lane1 & 0xFFu;

   sum4 = sum4 + cnt4;
   sum3 = sum3 + cnt3;
   sum2 = sum2 + cnt2;
   sum1 = sum1 + cnt1;


   //-
   // Accumulate sums from 2nd lane (of 4 groups)
   //-

   s3 = lane2 >> 16u;     // 3rd bin (of 4) in lane
   s2 = lane2 >>  8u;     // 2nd bin (of 4) in lane

   cnt4 = lane2 >> 24u;
   cnt3 = s3 & 0xFFu;
   cnt2 = s2 & 0xFFu;
   cnt1 = lane2 & 0xFFu;

   sum4 = sum4 + cnt4;
   sum3 = sum3 + cnt3;
   sum2 = sum2 + cnt2;
   sum1 = sum1 + cnt1;


   //-
   // Accumulate sums from 3rd lane (of 4 groups)
   //-

   s3 = lane3 >> 16u;     // 3rd bin (of 4) in lane
   s2 = lane3 >>  8u;     // 2nd bin (of 4) in lane

   cnt4 = lane3 >> 24u;
   cnt3 = s3 & 0xFFu;
   cnt2 = s2 & 0xFFu;
   cnt1 = lane3 & 0xFFu;

   sum4 = sum4 + cnt4;
   sum3 = sum3 + cnt3;
   sum2 = sum2 + cnt2;
   sum1 = sum1 + cnt1;


   //-
   // Accumulate sums from 4th lane (of 4 groups)
   //-

   s3 = lane4 >> 16u;     // 3rd bin (of 4) in lane
   s2 = lane4 >>  8u;     // 2nd bin (of 4) in lane

   cnt4 = lane4 >> 24u;
   cnt3 = s3 & 0xFFu;
   cnt2 = s2 & 0xFFu;
   cnt1 = lane4 & 0xFFu;

   sum4 = sum4 + cnt4;
   sum3 = sum3 + cnt3;
   sum2 = sum2 + cnt2;
   sum1 = sum1 + cnt1;
}


/*-------------------------------------------------------------------
  Name:   SS_Sums_4_Next_V2
  Desc:   Serial scan on next 4 elements in seq [0..3]
 ------------------------------------------------------------------*/

template < 
           U32 BlockSize,     // Threads Per Block
           U32 BlockMask      // Block Mask
         >
__device__ __forceinline__
void SS_Sums_4_Next_V2
( 
   U32 & sum13,    // OUT - sum1 .. sum4 (as pairs)
   U32 & sum24,
   U32 * cntPtr,   // IN  - 'per thread' counts <horizontal row> to sum up
   U32   baseIdx
) 
{
   // wrap = (idx + [0..3]) % BlockSize
   U32 idx1, idx2, idx3, idx4;
   idx1 = baseIdx + 0u;
   idx2 = baseIdx + 1u;
   idx3 = baseIdx + 2u;
   idx4 = baseIdx + 3u;

   U32 wrap1, wrap2, wrap3, wrap4;
   wrap1 = idx1 & BlockMask;
   wrap2 = idx2 & BlockMask;
   wrap3 = idx3 & BlockMask;
   wrap4 = idx4 & BlockMask;

   //-
   // Grab 4 elements in seq [0..3]
   //-

   U32 lane1, lane2, lane3, lane4;
   lane1 = cntPtr[wrap1];
   lane2 = cntPtr[wrap2];
   lane3 = cntPtr[wrap3];
   lane4 = cntPtr[wrap4];


   //-
   // Zero out sequence [0..3]
   //-

   cntPtr[wrap1] = 0u;
   cntPtr[wrap2] = 0u;
   cntPtr[wrap3] = 0u;
   cntPtr[wrap4] = 0u;


   //-
   // Accumulate all 4 groups in each lane
   //-

   //-
   // Initialize sums from 1st lane (of 4 groups)
   //-

   U32 cnt13, cnt24;
   cnt13 = (lane1 >> 0u) & 0x00FF00FFu;
   cnt24 = (lane1 >> 8u) & 0x00FF00FFu;
   sum13 += cnt13;
   sum24 += cnt24;


   //-
   // Accumulate sums from 2nd lane (of 4 groups)
   //-

   cnt13 = (lane2 >> 0u) & 0x00FF00FFu;
   cnt24 = (lane2 >> 8u) & 0x00FF00FFu;
   sum13 += cnt13;
   sum24 += cnt24;


   //-
   // Accumulate sums from 3rd lane (of 4 groups)
   //-

   cnt13 = (lane3 >> 0u) & 0x00FF00FFu;
   cnt24 = (lane3 >> 8u) & 0x00FF00FFu;
   sum13 += cnt13;
   sum24 += cnt24;


   //-
   // Accumulate sums from 4th lane (of 4 groups)
   //-

   cnt13 = (lane4 >> 0u) & 0x00FF00FFu;
   cnt24 = (lane4 >> 8u) & 0x00FF00FFu;
   sum13 += cnt13;
   sum24 += cnt24;
}


/*-------------------------------------------------------------------
  Name:   AddThreadToRowCounts_V1
  Desc:   Accumulates 'Per Thread' counts into 'Per Row' Counts
 ------------------------------------------------------------------*/

template < 
           U32 BlockSize,     // Threads per Block
           U32 BlockMask      // Block Mask
         >
__device__ __forceinline__
void AddThreadToRowCounts_V1
( 
   U32 & rCnt1,    // OUT - 4 'per row' counts assigned to this thread
   U32 & rCnt2,    //       ditto
   U32 & rCnt3,    //       ditto
   U32 & rCnt4,    //       ditto
   U32 * basePtr,  // IN  - array of 'per thread' counts
   U32   tid
) 
{
   //-----
   // Serial Scan (Scan All 64 elements in sequence)
   //-----

   // Accumulate [0..63]
      // Note: Also zeros out [0..63]
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid +  0) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid +  4) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid +  8) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 12) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 16) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 20) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 24) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 28) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 32) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 36) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 40) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 44) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 48) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 52) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 56) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 60) );
}


/*-------------------------------------------------------------------
  Name:   AddThreadToRowCounts_V2
  Desc:   Accumulates 'Per Thread' counts into 'Per Row' Counts
  Notes:   
  1. Vector Parallelism: 
       We accumulate 2 pairs at a time across each row 
       instead of 4 singletons for a big savings 
       in arithmetic operations.
  2. Overflow:
       We store 2 16-bit row sums per 32-bit number
       Which means that the accumulated Row sums need to not
       overflow a 16-bit number (65,535). 
       Since, we assume the maximum possible count per thread is 252
          64 threads * 252 =  16,128 <Safe>
         128 threads * 252 =  32,256 <Safe>
         256 threads * 252 =  64,512 <Safe>
         512 threads * 252 = 129,024 *** UNSAFE ***
       If this is a problem, revert to *_V1
  3. Register Pressure:
       *_V2 uses 6 more registers per thread than *_V1
       If this is a problem, revert to *_V1
 ------------------------------------------------------------------*/

template < 
           U32 BlockSize,     // Threads per Block
           U32 BlockMask      // BlockSize - 1
         >
__device__ __forceinline__
void AddThreadToRowCounts_V2
( 
   U32 & rCnt1,    // OUT - 4 'per row' counts assigned to this thread
   U32 & rCnt2,    //       ditto
   U32 & rCnt3,    //       ditto
   U32 & rCnt4,    //       ditto
   U32 * basePtr,  // IN  - array of 'per thread' counts
   U32   tid       // IN  - thread ID
) 
{
   U32 sum13, sum24;
   sum13 = 0u;
   sum24 = 0u;

   //-----
   // Serial Scan (Scan All 64 elements in sequence)
   //-----

   // Accumulate Row Sums [0..63]
      // Note: Also zeros out count array while accumulating
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid +  0) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid +  4) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid +  8) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 12) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 16) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 20) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 24) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 28) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 32) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 36) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 40) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 44) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 48) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 52) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 56) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 60) );

   // Convert row sums from pairs back into singletons
   U32 sum1, sum2, sum3, sum4;
   sum1 = sum13 & 0x0000FFFFu;
   sum2 = sum24 & 0x0000FFFFu;
   sum3 = sum13 >> 16u;
   sum4 = sum24 >> 16u;

   // Add row sums back into register counts
   rCnt1 += sum1;
   rCnt2 += sum2;
   rCnt3 += sum3;
   rCnt4 += sum4;
}


/*---------------------------------------------------------
  Name:   K1_TRISH_CountRows_GEN_B1
  Desc:   
  Note:   Assumes underlying data is stored as 
          four 8-bit values (U8,I8) per 32-bit 
          storage element
 ---------------------------------------------------------*/

template < 
           typename valT,	   // underlying value Type (U8, I8)
           typename mapT,	   // underlying mapper object 
           U32 logBankSize,	// log<2>( Channels per Bank )
           U32 logWarpSize,	// log<2>( Threads per Warp )
           U32 BlockSize,	   // Threads Per Block (needs to be a power of 2 & multiple of warpsize)
		     U32 GridSize,	   // Blocks Per Grid
           U32 K_length 	   // #elements to process per thread before looping
         >
__global__
void K1_TRISH_CountRows_GEN_B1
( 
         U32 * outRowCounts,	// OUT - 256-way row-sums array
   const U32 * inVals,			// IN  - values to bin and count
         U32   start,			// IN  - range [start,stop] to check and count
         U32   stop,			   //       ditto
         valT  minVal,			// IN  - minimum value
         valT  maxVal,			// IN  - maximum value
         U32   numBins        // IN  - number of bins (in histogram)
) 
{
   //-------------------------------------------
   // Constant values (computed at compile time)
   //-------------------------------------------

	   // Bank Size (elements per bank)
   const U32 BankSize    = (1u << logBankSize);	   // 32 = 2^5 threads per bank
   const U32 BankMask    = BankSize - 1u;	         // 31 = 32 - 1 = 0x1F = b11111
   const U32 strideBank  = BankSize + 1u;          // 33 = 32 + 1
      // Extra '+1' to help try and avoid bank conflicts

	   // Warp Size (threads per warp)
   const U32 WarpSize    = (1u << logWarpSize);	   // 32 = 2^5 threads per warp
   const U32 WarpMask    = WarpSize - 1u;			   // 31 = 32 - 1 = 0x1F = b11111

      // Block Size (threads per block)
   //const U32 BlockSize   = 64u;
   const U32 BlockMask   = BlockSize - 1u;

	   // Chunk Size
   //const U32 ChunkSize     = BlockSize * K_length;
   //const U32 IN_WarpSize   = K_length * WarpSize;

      // K_length
   //const U32 K_length = 16u;               // 16 
   const U32 K4_length = K_length * 4u;      // 64 = 16 * 4
   const U32 K4_stop   = 256u - K4_length;   // 192 = 256 - 64

	   // Warps Per Block
   const U32 WarpsPerBlock = BlockSize / WarpSize;   // 2 = 64/32

	   // Bins per Histogram
   const U32 nHistBins     = 256u;     // 256 = 2^8

	   // Lane Info (Compress 4 'bins' into each 32-bit value)
   const U32 nLanes		   = 64u;   // 64, # Lanes = 256 bins / 4 bins per lane

	   // 'Per Thread' counts array
   const U32 nTCounts      = nLanes * BlockSize;
   const U32 banksTCounts  = (nTCounts + BankMask) / BankSize;
   const U32 padTCounts    = (banksTCounts * BankSize) - nTCounts;
   const U32 sizeTCounts   = nTCounts + padTCounts;

      // Output size
   const U32 OutWarpSize   = nHistBins / WarpsPerBlock;
   const U32 OutLength     = OutWarpSize / WarpSize;
   const U32 OutStrideSize = OutLength * strideBank;

	   // Array Initialization
   const U32 nPassesThrd  = sizeTCounts / BlockSize;
   const U32 leftOverThrd = sizeTCounts - (nPassesThrd * BlockSize);

   const U32 nThreadsPerGrid = BlockSize * GridSize;	//   3,072 = 64 * 48
   const U32 rowSize = K_length * nThreadsPerGrid;		// 193,586 = 63 * 64 * 48


   //------------------------------------
   // Local Typedefs
   //------------------------------------

   // TRISH types
   typedef typename TRISH_traits<valT>::base_type    baseType;
   typedef typename TRISH_traits<valT>::bin_type     binType;
   typedef typename TRISH_traits<valT>::upscale_type upscaleType;
   typedef typename TRISH_traits<valT>::convert_type convertType;
   typedef typename ExtractBytes<upscaleType> Extractor;


   //------------------------------------
   // Local Variables
   //------------------------------------

	   // Local variables (shared memory)
   __shared__ U32  s_thrdCounts[sizeTCounts];   // 'per thread' counts

      // Local variables (registers)
   U32 rowCnt1 = 0u;
   U32 rowCnt2 = 0u;
   U32 rowCnt3 = 0u; 
   U32 rowCnt4 = 0u;


   //---------------------------
   // Compute Indices & Pointers
   //---------------------------

   U32 tid = threadIdx.x;		// Thread ID within Block
   U32 * cntPtr;
   U32 * basePtr;

   {
      // Get Warp Row & Column
      //U32 warpRow = threadIdx.x >> logWarpSize; // tid / 32
      //U32 warpCol = threadIdx.x & WarpMask;     // tid % 32

      // Compute starting 'input' offset (Warp Sequential Layout)
      //inIdx = (warpRow * IN_WarpSize) // Move to each warps assigned portion of work
      //        + warpCol;              // Move to warp column (in warp)

         // Compute starting serial scan index
      U32 baseIdx = (tid * BlockSize);

         // Get pointers into shared memory array
         // for different views of memory
      cntPtr  = &s_thrdCounts[threadIdx.x];
      basePtr = &s_thrdCounts[baseIdx];
   }


   //-------------------------------------------
   // Zero out arrays
   //-------------------------------------------

   {
	   //-
	   // Zero out 'Per Thread' counts
	   //-

      U32 * ptrTC = (&s_thrdCounts[0]);
      SetArray_BlockSeq
         < 
            U32, BlockSize, nPassesThrd, leftOverThrd, sizeTCounts
         >
         ( 
            ptrTC, 0u
         );
   }


   //-----
   // Compute thread, block, & grid indices & sizes
   //-----
 
   U32 bid = (blockIdx.y * gridDim.x) + blockIdx.x;		// Block ID within Grid
   U32 elemOffset = (bid * K_length * BlockSize) + tid;	// Starting offset 

   U32 nElems32        = stop - start + 1u;
   U32 nMaxRows        = (nElems32 + (rowSize - 1u)) / rowSize;
   U32 nSafeRows       = nElems32 / rowSize;
   U32 nSafeElems      = nSafeRows * rowSize;
   U32 nLeftOverElems  = nElems32 - nSafeElems;

   U32 startIdx        = start + elemOffset;
   U32 stopIdx         = startIdx + (nSafeRows * rowSize);
   U32 currIdx         = startIdx;
   U32 overflow        = 0u;

   // Initiate 

   // Initiate Mapping object 
   // (Transform from values to bin indices)
   mapT mapper;
   mapper.Initiate( minVal, maxVal, numBins );


   //-----
   // Process all safe blocks
   //-----

   // 'input' pointer for reading from memory
   const U32 * inPtr = &inVals[currIdx];

   // Sync Threads in Block
   if (WarpsPerBlock >= 2u) { __syncthreads(); }

   while (currIdx < stopIdx)
   {
      //-----
      // Accumulate 'thread' counts into 'row' counts
      //    Note: Also zeros out 'per thread' count array
      //-----

      if (overflow >= K4_stop)
      {
         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }

         //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         overflow = 0u;

         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }
      }

      U32         val1, val2, val3, val4;
      upscaleType b1, b2, b3, b4;
      binType     bin1, bin2, bin3, bin4;

         // NOTE:  the 'K_length' variable below is a static
         //        hard-coded constant in the range [1..63].
         //        K = 'Work per thread' per loop (stride)...
         //        The compiler will take care of throwing away 
         //        any unused code greater than our specified 'K'
         //        value, with no negative impact on performance.

      //-
      // Process values [0..3] (bytes 0..15)
      //-

      // Read in first 'four' values (32-bit)
      if (K_length >= 1u) { val1 = inPtr[0u*BlockSize]; }
      if (K_length >= 2u) { val2 = inPtr[1u*BlockSize]; }
      if (K_length >= 3u) { val3 = inPtr[2u*BlockSize]; }
      if (K_length >= 4u) { val4 = inPtr[3u*BlockSize]; }

      // Bin first 'four' values into count array
      if (K_length >= 1u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 2u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 3u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 4u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [4..7] (bytes 16..31)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 5u) { val1 = inPtr[4u*BlockSize]; }
      if (K_length >= 6u) { val2 = inPtr[5u*BlockSize]; }
      if (K_length >= 7u) { val3 = inPtr[6u*BlockSize]; }
      if (K_length >= 8u) { val4 = inPtr[7u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 5u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 6u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 7u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 8u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [8..11] (bytes 32..47)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >=  9u) { val1 = inPtr[ 8u*BlockSize]; } 
      if (K_length >= 10u) { val2 = inPtr[ 9u*BlockSize]; }
      if (K_length >= 11u) { val3 = inPtr[10u*BlockSize]; }
      if (K_length >= 12u) { val4 = inPtr[11u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 9u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 10u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 11u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 12u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [12..15] (bytes 48..63)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 13u) { val1 = inPtr[12u*BlockSize]; }
      if (K_length >= 14u) { val2 = inPtr[13u*BlockSize]; }
      if (K_length >= 15u) { val3 = inPtr[14u*BlockSize]; }
      if (K_length >= 16u) { val4 = inPtr[15u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 13u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 14u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 15u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 16u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [16..19] (bytes 64..79)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 17u) { val1 = inPtr[16u*BlockSize]; }
      if (K_length >= 18u) { val2 = inPtr[17u*BlockSize]; }
      if (K_length >= 19u) { val3 = inPtr[18u*BlockSize]; }
      if (K_length >= 20u) { val4 = inPtr[19u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 17u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 18u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 19u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 20u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [20..23] (bytes 80..95)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 21u) { val1 = inPtr[20u*BlockSize]; }
      if (K_length >= 22u) { val2 = inPtr[21u*BlockSize]; }
      if (K_length >= 23u) { val3 = inPtr[22u*BlockSize]; }
      if (K_length >= 24u) { val4 = inPtr[23u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 21u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 22u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 23u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 24u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [24..27] (bytes 96..111)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 25u) { val1 = inPtr[24u*BlockSize]; }
      if (K_length >= 26u) { val2 = inPtr[25u*BlockSize]; }
      if (K_length >= 27u) { val3 = inPtr[26u*BlockSize]; }
      if (K_length >= 28u) { val4 = inPtr[27u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 25u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 26u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 27u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 28u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [28..31] (bytes 112..127)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 29u) { val1 = inPtr[28u*BlockSize]; }
      if (K_length >= 30u) { val2 = inPtr[29u*BlockSize]; }
      if (K_length >= 31u) { val3 = inPtr[30u*BlockSize]; }
      if (K_length >= 32u) { val4 = inPtr[31u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      // Bin next 'four' values into count array
      if (K_length >= 29u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 30u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 31u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 32u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [32..35] (bytes 128..143)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 33u) { val1 = inPtr[32u*BlockSize]; }
      if (K_length >= 34u) { val2 = inPtr[33u*BlockSize]; }
      if (K_length >= 35u) { val3 = inPtr[34u*BlockSize]; }
      if (K_length >= 36u) { val4 = inPtr[35u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 33u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 34u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 35u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 36u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [36..39] (bytes 144..159)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 37u) { val1 = inPtr[36u*BlockSize]; }
      if (K_length >= 38u) { val2 = inPtr[37u*BlockSize]; }
      if (K_length >= 39u) { val3 = inPtr[38u*BlockSize]; }
      if (K_length >= 40u) { val4 = inPtr[39u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 37u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 38u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 39u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 40u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [40..43] (bytes 160-175)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 41u) { val1 = inPtr[40u*BlockSize]; }
      if (K_length >= 42u) { val2 = inPtr[41u*BlockSize]; }
      if (K_length >= 43u) { val3 = inPtr[42u*BlockSize]; }
      if (K_length >= 44u) { val4 = inPtr[43u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 41u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 42u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 43u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 44u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [44..47] (bytes 176-191)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 45u) { val1 = inPtr[44u*BlockSize]; }
      if (K_length >= 46u) { val2 = inPtr[45u*BlockSize]; }
      if (K_length >= 47u) { val3 = inPtr[46u*BlockSize]; }
      if (K_length >= 48u) { val4 = inPtr[47u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 45u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 46u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 47u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 48u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [48-51] (bytes 192-207)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 49u) { val1 = inPtr[48u*BlockSize]; }
      if (K_length >= 50u) { val2 = inPtr[49u*BlockSize]; }
      if (K_length >= 51u) { val3 = inPtr[50u*BlockSize]; }
      if (K_length >= 52u) { val4 = inPtr[51u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 49u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 50u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 51u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 52u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [52-55] (bytes 208-223)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 53u) { val1 = inPtr[52u*BlockSize]; }
      if (K_length >= 54u) { val2 = inPtr[53u*BlockSize]; }
      if (K_length >= 55u) { val3 = inPtr[54u*BlockSize]; }
      if (K_length >= 56u) { val4 = inPtr[55u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 53u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 54u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 55u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 56u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [56-59] (bytes 224-239)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 57u) { val1 = inPtr[56u*BlockSize]; }
      if (K_length >= 58u) { val2 = inPtr[57u*BlockSize]; }
      if (K_length >= 59u) { val3 = inPtr[58u*BlockSize]; }
      if (K_length >= 60u) { val4 = inPtr[59u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      // Bin next 'four' values into count array
      if (K_length >= 57u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 58u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 59u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 60u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }


      //-
      // Process values [60-62] (bytes 240-251)
      //-
	      // Note: We deliberately do not support k >= '64' to
	      //       avoid overflow issues during 'binning'
	      //       As our 'per thread' 'bin counts' can only handle 
	      //       '255' increments before overflow becomes a problem.
	      //       and 252 is the next smallest number 
	      //       evenly divisible by 4, IE 4 bytes per 32-bit value
	      //       63 values = 252 bytes / 4 bytes per value.

      // Read in next 'four' values (32-bit)
      if (K_length >= 61u) { val1 = inPtr[60u*BlockSize]; }
      if (K_length >= 62u) { val2 = inPtr[61u*BlockSize]; }
      if (K_length >= 63u) { val3 = inPtr[62u*BlockSize]; }

      // Note: Do not uncomment => *OVERFLOW* bug !!!
      //if (K_length >= 64u) { val4 = inPtr[63u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 60u) 
      { 
         Extractor::Extract4( b1, b2, b3, b4, val1 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 61u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      if (K_length >= 62u)
      {
         Extractor::Extract4( b1, b2, b3, b4, val3 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
 
      // Note: Do not uncomment => *OVERFLOW* bug !!!
      //if (K_length >= 63u)
      //{
      //   Extractor::Extract4( b1, b2, b3, b4, val4 );
	   //   mapper.Transform4( b1, b2, b3, b4, 
      //                      bin1, bin2, bin3, bin4 );
      //   BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      //}



      //-----
      // Move to next row of work
      //-----

      currIdx += rowSize;
        inPtr += rowSize;

      // Increment 'overflow' count
      overflow += K4_length;   // K values * 4 bytes per value
	}


   __syncthreads();

   //--------------------------------------
   // LAST: Process last leftover chunk
   //       with more careful range checking
   //--------------------------------------

   if (nLeftOverElems)
   {
      //-----
      // Accumulate 'thread' counts into 'row' counts
      //    Note: Also zeros out 'per thread' count array
      //-----

      if (overflow >= K4_stop)
      {
         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }

         //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         overflow = 0u;

         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }
      }

         // NOTE #1:  the 'K_length' variable below is a static
         //           hard-coded constant in the range [1..63].
         //           K = 'Work per thread' per loop (stride)...
         //           The compiler will take care of throwing away 
         //           any unused code greater than our specified 'K'
         //           value, with no negative impact on performance.

         // NOTE #2:  We use a cooperative stride 
         //           across each thread in each block in grid
         //           ChunkSize = BlockSize * GridSize = 64 * 48 = 3072
         //           RowSize   = WorkPerThead(K) * ChunkSize = 63 * 3072 = 193,536
         // 
         //                       B0   B1  ...  B47  (Blocks in Grid)
         //                      ---- ---- --- ----
         //           k =  1 =>  |64| |64| ... |64|  (3072 Thread & I/O requests for 1st work item per thread)
         //           k =  2 =>  |64| |64| ... |64|  ditto (2nd work item per thread)
         //               ...       ...         ...
         //           k = 63 =>  |64| |64| ... |64|  ditto (63 work item per thread)

         // NOTE #3:  We use "Divide & Conquer" to avoid as much slower range checking as possible
         //			  Try batches of 32, 16, 8, 4, 2, 1, and finally leftover (on which we finally must range check) 

      //----
      // Setup Pointers & Indices for cooperative stride 
      //----

      U32 bid        = (blockIdx.y * gridDim.x) + blockIdx.x;	// Get block index
      U32 nSkip      = nSafeRows * rowSize;						   // Skip past already processed rows
      U32 chunkIdx   = (bid * BlockSize) + tid;					   // Get starting index within chunk
      U32 baseIdx    = start + nSkip + chunkIdx;				   // Get starting index for left over elements

      U32         val1, val2, val3, val4;
      upscaleType b1, b2, b3, b4;
      binType     bin1, bin2, bin3, bin4;


      //------
      // Try Section of 32
      //------

      if (K_length >= 32u)
      {
         // Process 32 chunks safely without range checking
         if (nLeftOverElems >= (32u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [9..12]
            //-----

            val1 = inPtr[( 8u*nThreadsPerGrid)];
            val2 = inPtr[( 9u*nThreadsPerGrid)];
            val3 = inPtr[(10u*nThreadsPerGrid)];
            val4 = inPtr[(11u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [13..16]
            //-----

            val1 = inPtr[(12u*nThreadsPerGrid)];
            val2 = inPtr[(13u*nThreadsPerGrid)];
            val3 = inPtr[(14u*nThreadsPerGrid)];
            val4 = inPtr[(15u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [17..20]
            //-----

            val1 = inPtr[(16u*nThreadsPerGrid)];
            val2 = inPtr[(17u*nThreadsPerGrid)];
            val3 = inPtr[(18u*nThreadsPerGrid)];
            val4 = inPtr[(19u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [21..24]
            //-----

            val1 = inPtr[(20u*nThreadsPerGrid)];
            val2 = inPtr[(21u*nThreadsPerGrid)];
            val3 = inPtr[(22u*nThreadsPerGrid)];
            val4 = inPtr[(23u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [25..28]
            //-----

            val1 = inPtr[(24u*nThreadsPerGrid)];
            val2 = inPtr[(25u*nThreadsPerGrid)];
            val3 = inPtr[(26u*nThreadsPerGrid)];
            val4 = inPtr[(27u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [29..32]
            //-----

            val1 = inPtr[(28u*nThreadsPerGrid)];
            val2 = inPtr[(29u*nThreadsPerGrid)];
            val3 = inPtr[(30u*nThreadsPerGrid)];
            val4 = inPtr[(31u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
            
            // Move to next section
            baseIdx        += (32u * nThreadsPerGrid);
            nLeftOverElems -= (32u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 16
      //------

      if (K_length >= 16u)
      {
         // Process 16 chunks safely without range checking
         if (nLeftOverElems >= (16u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [9..12]
            //-----

            val1 = inPtr[( 8u*nThreadsPerGrid)];
            val2 = inPtr[( 9u*nThreadsPerGrid)];
            val3 = inPtr[(10u*nThreadsPerGrid)];
            val4 = inPtr[(11u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [13..16]
            //-----

            val1 = inPtr[(12u*nThreadsPerGrid)];
            val2 = inPtr[(13u*nThreadsPerGrid)];
            val3 = inPtr[(14u*nThreadsPerGrid)];
            val4 = inPtr[(15u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (16u * nThreadsPerGrid);
            nLeftOverElems -= (16u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 8
      //------

      if (K_length >= 8u)
      {
         // Process 8 chunks safely without range checking
         if (nLeftOverElems >= (8u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (8u * nThreadsPerGrid);
            nLeftOverElems -= (8u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 4
      //------

      if (K_length >= 4u)
      {
         // Process 4 chunks safely without range checking
         if (nLeftOverElems >= (4u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val3 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (4u * nThreadsPerGrid);
            nLeftOverElems -= (4u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 2
      //------

      if (K_length >= 2u)
      {
         // Process 2 chunks safely without range checking
         if (nLeftOverElems >= (2u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..2]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            Extractor::Extract4( b1, b2, b3, b4, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (2u * nThreadsPerGrid);
            nLeftOverElems -= (2u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 1
      //------

      if (K_length >= 1u)
      {
         // Process 1 chunk safely without range checking
         if (nLeftOverElems >= (1u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];

            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Move to next section
            baseIdx        += (1u * nThreadsPerGrid);
            nLeftOverElems -= (1u * nThreadsPerGrid);
         }
      }


      //------
      // Process Last few elements
      //    with careful RANGE CHECKING !!!
      //------

      if (nLeftOverElems > 0u)
      {
         // Make sure we are 'in range' before reading & binning
         U32 inRange1 = (baseIdx <= stop);
         if (inRange1) 
         { 
            // Read in 32-bit element
            val1 = inVals[baseIdx];

            // Process element
            Extractor::Extract4( b1, b2, b3, b4, val1 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
      }

      // Update Accumulation count
      overflow += K4_length;   // 64 = 16 elems * 4 bytes per elem
	}


	// Cleanup Mapping object 
	// (Give mapper a chance to cleanup any resources)
	mapper.Finish();


   //-----
   // Accumulate 'thread' counts into 'row' counts
   //    Note: Also zeros out 'per thread' count array
   //-----

   if (overflow > 0u)
   {
      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }

      //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
      AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
      overflow = 0u;

      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }
   }



	//-------------------------------------------------
	// Write out final row 'counts'
	//-------------------------------------------------

   {
      // Compute starting 'row counts' offset
      U32 rIdx = threadIdx.x * 4u;         // 4 groups per lane
      U32 rRow = rIdx >> logBankSize;
      U32 rCol = rIdx & BankMask;

      U32 rowIdx = (rRow * strideBank) + (rCol + 1u);
         // Extra '+1' to shift past initial pad element      

      U32 * rowPtr = &s_thrdCounts[rowIdx];

      // Store row counts in row array
      rowPtr[0] = rowCnt1;
      rowPtr[1] = rowCnt2;
      rowPtr[2] = rowCnt3;
      rowPtr[3] = rowCnt4;

      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }

      // Get Warp Row & Column
      U32 warpRow = threadIdx.x >> logWarpSize; // tid / 32
      U32 warpCol = threadIdx.x & WarpMask;     // tid % 32

      // Get local & global indices
      U32 outGlobal = (blockIdx.x * nHistBins);
      U32 outLocal  = (warpRow * OutWarpSize);
      U32 rowBase   = (warpRow * OutStrideSize);
      U32 outBase   = outGlobal + outLocal;
      U32 rowOff    = warpCol + 1u;

      U32 outIdx = outBase + warpCol;
          rowIdx = rowBase + rowOff;

      // Get local & global pointers
      U32 * outPtr = &outRowCounts[outIdx];
            rowPtr = &s_thrdCounts[rowIdx];

         // Write our 'per row' counts in warp sequential order
      if (OutLength >= 1u) { outPtr[(0u*WarpSize)] = rowPtr[(0u*strideBank)]; }
      if (OutLength >= 2u) { outPtr[(1u*WarpSize)] = rowPtr[(1u*strideBank)]; }
      if (OutLength >= 3u) { outPtr[(2u*WarpSize)] = rowPtr[(2u*strideBank)]; }
      if (OutLength >= 4u) { outPtr[(3u*WarpSize)] = rowPtr[(3u*strideBank)]; }
      if (OutLength >= 5u) { outPtr[(4u*WarpSize)] = rowPtr[(4u*strideBank)]; }
      if (OutLength >= 6u) { outPtr[(5u*WarpSize)] = rowPtr[(5u*strideBank)]; }
      if (OutLength >= 7u) { outPtr[(6u*WarpSize)] = rowPtr[(6u*strideBank)]; }
      if (OutLength >= 8u) { outPtr[(7u*WarpSize)] = rowPtr[(7u*strideBank)]; }
   }
}


/*---------------------------------------------------------
  Name:   K1_TRISH_CountRows_GEN_B2
  Desc:   
  Note:   
  
  1. Assumes underlying data is stored as two 16-bit values 
    (U16,I16) per 32-bit storage element.
  2. This further implies that K = [1,127] to safely
     avoid overflowing an 8-bit counter.
 ---------------------------------------------------------*/

template < 
           typename valT,	   // underlying value Type (U8, I8)
           typename mapT,	   // underlying mapper object 
           U32 logBankSize,	// log<2>( Channels per Bank )
           U32 logWarpSize,	// log<2>( Threads per Warp )
           U32 BlockSize,	   // Threads Per Block (needs to be a power of 2 & multiple of warpsize)
		     U32 GridSize,	   // Blocks Per Grid
           U32 K_length 	   // #elements to process per thread before looping
         >
__global__
void K1_TRISH_CountRows_GEN_B2
( 
         U32 * outRowCounts,	// OUT - 256-way row-sums array
   const U32 * inVals,			// IN  - values to bin and count
         U32   start,			// IN  - range [start,stop] to check and count
         U32   stop,			   //       ditto
         valT  minVal,			// IN  - minimum value
         valT  maxVal,			// IN  - maximum value
         U32   numBins        // IN  - number of bins (in histogram)
) 
{
   //-------------------------------------------
   // Constant values (computed at compile time)
   //-------------------------------------------

	   // Bank Size (elements per bank)
   const U32 BankSize    = (1u << logBankSize);	   // 32 = 2^5 threads per bank
   const U32 BankMask    = BankSize - 1u;	         // 31 = 32 - 1 = 0x1F = b11111
   const U32 strideBank  = BankSize + 1u;          // 33 = 32 + 1
      // Extra '+1' to help try and avoid bank conflicts

	   // Warp Size (threads per warp)
   const U32 WarpSize    = (1u << logWarpSize);	   // 32 = 2^5 threads per warp
   const U32 WarpMask    = WarpSize - 1u;			   // 31 = 32 - 1 = 0x1F = b11111

      // Block Size (threads per block)
   //const U32 BlockSize   = 64u;
   const U32 BlockMask   = BlockSize - 1u;

	   // Chunk Size
   //const U32 ChunkSize     = BlockSize * K_length;
   //const U32 IN_WarpSize   = K_length * WarpSize;

      // K_length
   //const U32 K_length = 16u;               //  16 
   const U32 K2_length = K_length * 2u;      //  32 = 16 * 2  (2 words per 32-bit input value)
   const U32 K2_stop   = 256u - K2_length;   // 224 = 256 - 32 (conservative test)

	   // Warps Per Block
   const U32 WarpsPerBlock = BlockSize / WarpSize;   // 2 = 64/32

	   // Bins per Histogram
   const U32 nHistBins     = 256u;     // 256 = 2^8

	   // Lane Info (Compress 4 'bins' into each 32-bit value)
   const U32 nLanes		   = 64u;   // 64, # Lanes = 256 bins / 4 bins per lane

	   // 'Per Thread' counts array
   const U32 nTCounts      = nLanes * BlockSize;
   const U32 banksTCounts  = (nTCounts + BankMask) / BankSize;
   const U32 padTCounts    = (banksTCounts * BankSize) - nTCounts;
   const U32 sizeTCounts   = nTCounts + padTCounts;

      // Output size
   const U32 OutWarpSize   = nHistBins / WarpsPerBlock;
   const U32 OutLength     = OutWarpSize / WarpSize;
   const U32 OutStrideSize = OutLength * strideBank;

	   // Array Initialization
   const U32 nPassesThrd  = sizeTCounts / BlockSize;
   const U32 leftOverThrd = sizeTCounts - (nPassesThrd * BlockSize);

   const U32 nThreadsPerGrid = BlockSize * GridSize;	//   3,072 = 64 * 48
   const U32 rowSize = K_length * nThreadsPerGrid;		// 193,586 = 63 * 64 * 48


   //------------------------------------
   // Local Typedefs
   //------------------------------------

   // TRISH types
   typedef typename TRISH_traits<valT>::base_type    baseType;
   typedef typename TRISH_traits<valT>::bin_type     binType;
   typedef typename TRISH_traits<valT>::upscale_type upscaleType;
   typedef typename TRISH_traits<valT>::convert_type convertType;
   typedef typename ExtractWords<upscaleType> Extractor;


   //------------------------------------
   // Local Variables
   //------------------------------------

	   // Local variables (shared memory)
   __shared__ U32  s_thrdCounts[sizeTCounts];   // 'per thread' counts

      // Local variables (registers)
   U32 rowCnt1 = 0u;
   U32 rowCnt2 = 0u;
   U32 rowCnt3 = 0u; 
   U32 rowCnt4 = 0u;


   //---------------------------
   // Compute Indices & Pointers
   //---------------------------

   U32 tid = threadIdx.x;		// Thread ID within Block
   U32 * cntPtr;
   U32 * basePtr;

   {
      // Get Warp Row & Column
      //U32 warpRow = threadIdx.x >> logWarpSize; // tid / 32
      //U32 warpCol = threadIdx.x & WarpMask;     // tid % 32

      // Compute starting 'input' offset (Warp Sequential Layout)
      //inIdx = (warpRow * IN_WarpSize) // Move to each warps assigned portion of work
      //        + warpCol;              // Move to warp column (in warp)

         // Compute starting serial scan index
      U32 baseIdx = (tid * BlockSize);

         // Get pointers into shared memory array
         // for different views of memory
      cntPtr  = &s_thrdCounts[threadIdx.x];
      basePtr = &s_thrdCounts[baseIdx];
   }


   //-------------------------------------------
   // Zero out arrays
   //-------------------------------------------

   {
	   //-
	   // Zero out 'Per Thread' counts
	   //-

      U32 * ptrTC = (&s_thrdCounts[0]);
      SetArray_BlockSeq
         < 
            U32, BlockSize, nPassesThrd, leftOverThrd, sizeTCounts
         >
         ( 
            ptrTC, 0u
         );
   }


   //-----
   // Compute thread, block, & grid indices & sizes
   //-----
 
   U32 bid = (blockIdx.y * gridDim.x) + blockIdx.x;		// Block ID within Grid
   U32 elemOffset = (bid * K_length * BlockSize) + tid;	// Starting offset 

   U32 nElems32        = stop - start + 1u;
   U32 nMaxRows        = (nElems32 + (rowSize - 1u)) / rowSize;
   U32 nSafeRows       = nElems32 / rowSize;
   U32 nSafeElems      = nSafeRows * rowSize;
   U32 nLeftOverElems  = nElems32 - nSafeElems;

   U32 startIdx        = start + elemOffset;
   U32 stopIdx         = startIdx + (nSafeRows * rowSize);
   U32 currIdx         = startIdx;
   U32 overflow        = 0u;

   // Initiate 

   // Initiate Mapping object 
   // (Transform from values to bin indices)
   mapT mapper;
   mapper.Initiate( minVal, maxVal, numBins );


   //-----
   // Process all safe blocks
   //-----

   // 'input' pointer for reading from memory
   const U32 * inPtr = &inVals[currIdx];

   // Sync Threads in Block
   if (WarpsPerBlock >= 2u) { __syncthreads(); }

   while (currIdx < stopIdx)
   {
      //-----
      // Accumulate 'thread' counts into 'row' counts
      //    Note: Also zeros out 'per thread' count array
      //-----

      if (overflow >= K2_stop)
      {
         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }

         //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         overflow = 0u;

         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }
      }

      U32         val1, val2, val3, val4;
      upscaleType b1, b2, b3, b4;
      binType     bin1, bin2, bin3, bin4;

         // NOTE:  the 'K_length' variable below is a static
         //        hard-coded constant in the range [1..63].
         //        K = 'Work per thread' per loop (stride)...
         //        The compiler will take care of throwing away 
         //        any unused code greater than our specified 'K'
         //        value, with no negative impact on performance.

      //-
      // Process values [0..3] (bytes 0..15)
      //-

      // Read in first 'four' values (32-bit)
      if (K_length >= 1u) { val1 = inPtr[0u*BlockSize]; }
      if (K_length >= 2u) { val2 = inPtr[1u*BlockSize]; }
      if (K_length >= 3u) { val3 = inPtr[2u*BlockSize]; }
      if (K_length >= 4u) { val4 = inPtr[3u*BlockSize]; }

      // Bin first 'four' values into count array
      if (K_length >= 4u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 3u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 2u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 1u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [4..7] (bytes 16..31)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 5u) { val1 = inPtr[4u*BlockSize]; }
      if (K_length >= 6u) { val2 = inPtr[5u*BlockSize]; }
      if (K_length >= 7u) { val3 = inPtr[6u*BlockSize]; }
      if (K_length >= 8u) { val4 = inPtr[7u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 8u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 7u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1, b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 6u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 5u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [8..11] (bytes 32..47)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >=  9u) { val1 = inPtr[ 8u*BlockSize]; } 
      if (K_length >= 10u) { val2 = inPtr[ 9u*BlockSize]; }
      if (K_length >= 11u) { val3 = inPtr[10u*BlockSize]; }
      if (K_length >= 12u) { val4 = inPtr[11u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 12u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 11u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 10u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 9u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [12..15] (bytes 48..63)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 13u) { val1 = inPtr[12u*BlockSize]; }
      if (K_length >= 14u) { val2 = inPtr[13u*BlockSize]; }
      if (K_length >= 15u) { val3 = inPtr[14u*BlockSize]; }
      if (K_length >= 16u) { val4 = inPtr[15u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 16u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 15u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 14u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 13u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [16..19] (bytes 64..79)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 17u) { val1 = inPtr[16u*BlockSize]; }
      if (K_length >= 18u) { val2 = inPtr[17u*BlockSize]; }
      if (K_length >= 19u) { val3 = inPtr[18u*BlockSize]; }
      if (K_length >= 20u) { val4 = inPtr[19u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 20u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 19u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 18u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 17u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [20..23] (bytes 80..95)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 21u) { val1 = inPtr[20u*BlockSize]; }
      if (K_length >= 22u) { val2 = inPtr[21u*BlockSize]; }
      if (K_length >= 23u) { val3 = inPtr[22u*BlockSize]; }
      if (K_length >= 24u) { val4 = inPtr[23u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 24u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 23u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 22u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 21u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [24..27] (bytes 96..111)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 25u) { val1 = inPtr[24u*BlockSize]; }
      if (K_length >= 26u) { val2 = inPtr[25u*BlockSize]; }
      if (K_length >= 27u) { val3 = inPtr[26u*BlockSize]; }
      if (K_length >= 28u) { val4 = inPtr[27u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 28u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 27u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 26u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 25u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [28..31] (bytes 112..127)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 29u) { val1 = inPtr[28u*BlockSize]; }
      if (K_length >= 30u) { val2 = inPtr[29u*BlockSize]; }
      if (K_length >= 31u) { val3 = inPtr[30u*BlockSize]; }
      if (K_length >= 32u) { val4 = inPtr[31u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 32u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 31u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 30u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 29u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [32..35] (bytes 128..143)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 33u) { val1 = inPtr[32u*BlockSize]; }
      if (K_length >= 34u) { val2 = inPtr[33u*BlockSize]; }
      if (K_length >= 35u) { val3 = inPtr[34u*BlockSize]; }
      if (K_length >= 36u) { val4 = inPtr[35u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 36u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 35u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 34u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 33u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [36..39] (bytes 144..159)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 37u) { val1 = inPtr[36u*BlockSize]; }
      if (K_length >= 38u) { val2 = inPtr[37u*BlockSize]; }
      if (K_length >= 39u) { val3 = inPtr[38u*BlockSize]; }
      if (K_length >= 40u) { val4 = inPtr[39u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 40u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 39u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 38u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 37u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [40..43] (bytes 160-175)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 41u) { val1 = inPtr[40u*BlockSize]; }
      if (K_length >= 42u) { val2 = inPtr[41u*BlockSize]; }
      if (K_length >= 43u) { val3 = inPtr[42u*BlockSize]; }
      if (K_length >= 44u) { val4 = inPtr[43u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 44u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 43u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 42u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 41u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [44..47] (bytes 176-191)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 45u) { val1 = inPtr[44u*BlockSize]; }
      if (K_length >= 46u) { val2 = inPtr[45u*BlockSize]; }
      if (K_length >= 47u) { val3 = inPtr[46u*BlockSize]; }
      if (K_length >= 48u) { val4 = inPtr[47u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 48u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 47u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 46u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 45u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [48-51] (bytes 192-207)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 49u) { val1 = inPtr[48u*BlockSize]; }
      if (K_length >= 50u) { val2 = inPtr[49u*BlockSize]; }
      if (K_length >= 51u) { val3 = inPtr[50u*BlockSize]; }
      if (K_length >= 52u) { val4 = inPtr[51u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 52u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 51u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 50u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 49u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [52-55] (bytes 208-223)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 53u) { val1 = inPtr[52u*BlockSize]; }
      if (K_length >= 54u) { val2 = inPtr[53u*BlockSize]; }
      if (K_length >= 55u) { val3 = inPtr[54u*BlockSize]; }
      if (K_length >= 56u) { val4 = inPtr[55u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 56u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 55u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 54u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 53u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [56-59] (bytes 224-239)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 57u) { val1 = inPtr[56u*BlockSize]; }
      if (K_length >= 58u) { val2 = inPtr[57u*BlockSize]; }
      if (K_length >= 59u) { val3 = inPtr[58u*BlockSize]; }
      if (K_length >= 60u) { val4 = inPtr[59u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 60u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 59u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 58u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                               b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 57u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [60-63]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 61u) { val1 = inPtr[60u*BlockSize]; }
      if (K_length >= 62u) { val2 = inPtr[61u*BlockSize]; }
      if (K_length >= 63u) { val3 = inPtr[62u*BlockSize]; }
      if (K_length >= 64u) { val4 = inPtr[63u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 64u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 63u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 62u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 61u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [64-67]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 65u) { val1 = inPtr[64u*BlockSize]; }
      if (K_length >= 66u) { val2 = inPtr[65u*BlockSize]; }
      if (K_length >= 67u) { val3 = inPtr[66u*BlockSize]; }
      if (K_length >= 68u) { val4 = inPtr[67u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 68u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 67u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 66u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 65u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [68-71]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 69u) { val1 = inPtr[68u*BlockSize]; }
      if (K_length >= 70u) { val2 = inPtr[69u*BlockSize]; }
      if (K_length >= 71u) { val3 = inPtr[70u*BlockSize]; }
      if (K_length >= 72u) { val4 = inPtr[71u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 72u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 71u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 70u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 69u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [72-75]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 73u) { val1 = inPtr[72u*BlockSize]; }
      if (K_length >= 74u) { val2 = inPtr[73u*BlockSize]; }
      if (K_length >= 75u) { val3 = inPtr[74u*BlockSize]; }
      if (K_length >= 76u) { val4 = inPtr[75u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 76u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 75u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 74u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 73u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [76-79]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 77u) { val1 = inPtr[76u*BlockSize]; }
      if (K_length >= 78u) { val2 = inPtr[77u*BlockSize]; }
      if (K_length >= 79u) { val3 = inPtr[78u*BlockSize]; }
      if (K_length >= 80u) { val4 = inPtr[79u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 80u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 79u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 78u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 77u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [80-83]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 81u) { val1 = inPtr[80u*BlockSize]; }
      if (K_length >= 82u) { val2 = inPtr[81u*BlockSize]; }
      if (K_length >= 83u) { val3 = inPtr[82u*BlockSize]; }
      if (K_length >= 84u) { val4 = inPtr[83u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 84u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 83u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 82u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 81u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [84-87]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 85u) { val1 = inPtr[84u*BlockSize]; }
      if (K_length >= 86u) { val2 = inPtr[85u*BlockSize]; }
      if (K_length >= 87u) { val3 = inPtr[86u*BlockSize]; }
      if (K_length >= 88u) { val4 = inPtr[87u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 88u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 87u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 86u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 85u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [88-91]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 89u) { val1 = inPtr[88u*BlockSize]; }
      if (K_length >= 90u) { val2 = inPtr[89u*BlockSize]; }
      if (K_length >= 91u) { val3 = inPtr[90u*BlockSize]; }
      if (K_length >= 92u) { val4 = inPtr[91u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 92u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 91u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 90u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 89u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [92-95]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 93u) { val1 = inPtr[92u*BlockSize]; }
      if (K_length >= 94u) { val2 = inPtr[93u*BlockSize]; }
      if (K_length >= 95u) { val3 = inPtr[94u*BlockSize]; }
      if (K_length >= 96u) { val4 = inPtr[95u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 96u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 95u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 94u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 93u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [96-99]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >=  97u) { val1 = inPtr[96u*BlockSize]; }
      if (K_length >=  98u) { val2 = inPtr[97u*BlockSize]; }
      if (K_length >=  99u) { val3 = inPtr[98u*BlockSize]; }
      if (K_length >= 100u) { val4 = inPtr[99u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 100u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 99u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 98u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 97u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [100-103]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 101u) { val1 = inPtr[100u*BlockSize]; }
      if (K_length >= 102u) { val2 = inPtr[101u*BlockSize]; }
      if (K_length >= 103u) { val3 = inPtr[102u*BlockSize]; }
      if (K_length >= 104u) { val4 = inPtr[103u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 104u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 103u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 102u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 101u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [104-107]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 105u) { val1 = inPtr[104u*BlockSize]; }
      if (K_length >= 106u) { val2 = inPtr[105u*BlockSize]; }
      if (K_length >= 107u) { val3 = inPtr[106u*BlockSize]; }
      if (K_length >= 108u) { val4 = inPtr[107u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 108u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 107u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 106u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 105u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [108-111]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 109u) { val1 = inPtr[108u*BlockSize]; }
      if (K_length >= 110u) { val2 = inPtr[109u*BlockSize]; }
      if (K_length >= 111u) { val3 = inPtr[110u*BlockSize]; }
      if (K_length >= 112u) { val4 = inPtr[111u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 112u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 111u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 110u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 109u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [112-115]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 113u) { val1 = inPtr[112u*BlockSize]; }
      if (K_length >= 114u) { val2 = inPtr[113u*BlockSize]; }
      if (K_length >= 115u) { val3 = inPtr[114u*BlockSize]; }
      if (K_length >= 116u) { val4 = inPtr[115u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 116u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 115u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 114u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 113u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [116-119]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 117u) { val1 = inPtr[116u*BlockSize]; }
      if (K_length >= 118u) { val2 = inPtr[117u*BlockSize]; }
      if (K_length >= 119u) { val3 = inPtr[118u*BlockSize]; }
      if (K_length >= 120u) { val4 = inPtr[119u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 120u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 119u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 118u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 117u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [120-123]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 121u) { val1 = inPtr[120u*BlockSize]; }
      if (K_length >= 122u) { val2 = inPtr[121u*BlockSize]; }
      if (K_length >= 123u) { val3 = inPtr[122u*BlockSize]; }
      if (K_length >= 124u) { val4 = inPtr[123u*BlockSize]; }

      // Bin next 'four' values into count array
      if (K_length >= 124u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      {
         if (K_length == 123u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 122u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 121u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-
      // Process values [124-127]
      //-

      // Read in next 'three' values (32-bit)
      if (K_length >= 125u) { val1 = inPtr[124u*BlockSize]; }
      if (K_length >= 126u) { val2 = inPtr[125u*BlockSize]; }
      if (K_length >= 127u) { val3 = inPtr[126u*BlockSize]; }
      
      // NOTE: Do not uncomment the line below => *OVERFLOW* BUG !!!
      //if (K_length >= 128u) { val4 = inPtr[127u*BlockSize]; }

      // Bin next 'four' values into count array

      // NOTE: Do not uncomment the section below => *OVERFLOW* BUG !!!
      /*
      if (K_length >= 128u) 
      { 
         // Process v1,v2
         Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

         // Process v3,v4
         Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else
      */
      {
         if (K_length == 127u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3
            Extractor::Extract2( b1, b2, val3 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 126u)
         {
            // Process v1,v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
	         mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                                 b1,   b2,   b3,   b4 ); // IN => values to transform
	         BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
         }
         if (K_length == 125u)
         {
            // Process v1
            Extractor::Extract2( b1, b2, val1 );
	         mapper.Transform2( bin1, bin2, b1,   b2 );
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }


      //-----
      // Move to next row of work
      //-----

      currIdx += rowSize;
        inPtr += rowSize;

      // Increment 'overflow' count
      overflow += K2_length;   // K values * 2 words per value
	}


   __syncthreads();

   //--------------------------------------
   // LAST: Process last leftover chunk
   //       with more careful range checking
   //--------------------------------------

   if (nLeftOverElems)
   {
      //-----
      // Accumulate 'thread' counts into 'row' counts
      //    Note: Also zeros out 'per thread' count array
      //-----

      if (overflow >= K2_stop)
      {
         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }

         //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         overflow = 0u;

         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }
      }

         // NOTE #1:  the 'K_length' variable below is a static
         //           hard-coded constant in the range [1..127].
         //           K = 'Work per thread' per loop (stride)...
         //           The compiler will take care of throwing away 
         //           any unused code greater than our specified 'K'
         //           value, with no negative impact on performance.

         // NOTE #2:  We use a cooperative stride 
         //           across each thread in each block in grid
         //           ChunkSize = BlockSize * GridSize = 64 * 48 = 3072
         //           RowSize   = WorkPerThead(K) * ChunkSize = 63 * 3072 = 193,536
         // 
         //                       B0   B1  ...  B47  (Blocks in Grid)
         //                      ---- ---- --- ----
         //           k =  1 =>  |64| |64| ... |64|  (3072 Thread & I/O requests for 1st work item per thread)
         //           k =  2 =>  |64| |64| ... |64|  ditto (2nd work item per thread)
         //               ...       ...         ...
         //           k = 63 =>  |64| |64| ... |64|  ditto (63 work item per thread)

         // NOTE #3:  We use "Divide & Conquer" to avoid as much slower range checking as possible
         //			  Try batches of 32, 16, 8, 4, 2, 1, and finally leftover (on which we finally must range check) 

      //----
      // Setup Pointers & Indices for cooperative stride 
      //----

      U32 bid        = (blockIdx.y * gridDim.x) + blockIdx.x;	// Get block index
      U32 nSkip      = nSafeRows * rowSize;						   // Skip past already processed rows
      U32 chunkIdx   = (bid * BlockSize) + tid;					   // Get starting index within chunk
      U32 baseIdx    = start + nSkip + chunkIdx;				   // Get starting index for left over elements

      U32         val1, val2, val3, val4;
      upscaleType b1, b2, b3, b4;
      binType     bin1, bin2, bin3, bin4;

      //------
      // Try Section of 64
      //------

      if (K_length >= 64u)
      {
         // Process 64 chunks safely without range checking
         if (nLeftOverElems >= (64u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [9..12]
            //-----

            val1 = inPtr[( 8u*nThreadsPerGrid)];
            val2 = inPtr[( 9u*nThreadsPerGrid)];
            val3 = inPtr[(10u*nThreadsPerGrid)];
            val4 = inPtr[(11u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [13..16]
            //-----

            val1 = inPtr[(12u*nThreadsPerGrid)];
            val2 = inPtr[(13u*nThreadsPerGrid)];
            val3 = inPtr[(14u*nThreadsPerGrid)];
            val4 = inPtr[(15u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [17..20]
            //-----

            val1 = inPtr[(16u*nThreadsPerGrid)];
            val2 = inPtr[(17u*nThreadsPerGrid)];
            val3 = inPtr[(18u*nThreadsPerGrid)];
            val4 = inPtr[(19u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [21..24]
            //-----

            val1 = inPtr[(20u*nThreadsPerGrid)];
            val2 = inPtr[(21u*nThreadsPerGrid)];
            val3 = inPtr[(22u*nThreadsPerGrid)];
            val4 = inPtr[(23u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [25..28]
            //-----

            val1 = inPtr[(24u*nThreadsPerGrid)];
            val2 = inPtr[(25u*nThreadsPerGrid)];
            val3 = inPtr[(26u*nThreadsPerGrid)];
            val4 = inPtr[(27u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [29..32]
            //-----

            val1 = inPtr[(28u*nThreadsPerGrid)];
            val2 = inPtr[(29u*nThreadsPerGrid)];
            val3 = inPtr[(30u*nThreadsPerGrid)];
            val4 = inPtr[(31u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [33..35]
            //-----

            val1 = inPtr[(32u*nThreadsPerGrid)];
            val2 = inPtr[(33u*nThreadsPerGrid)];
            val3 = inPtr[(34u*nThreadsPerGrid)];
            val4 = inPtr[(35u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [36..39]
            //-----

            val1 = inPtr[(36u*nThreadsPerGrid)];
            val2 = inPtr[(37u*nThreadsPerGrid)];
            val3 = inPtr[(38u*nThreadsPerGrid)];
            val4 = inPtr[(39u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [40..43]
            //-----

            val1 = inPtr[(40u*nThreadsPerGrid)];
            val2 = inPtr[(41u*nThreadsPerGrid)];
            val3 = inPtr[(42u*nThreadsPerGrid)];
            val4 = inPtr[(43u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [44..47]
            //-----

            val1 = inPtr[(44u*nThreadsPerGrid)];
            val2 = inPtr[(45u*nThreadsPerGrid)];
            val3 = inPtr[(46u*nThreadsPerGrid)];
            val4 = inPtr[(47u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [48..51]
            //-----

            val1 = inPtr[(48u*nThreadsPerGrid)];
            val2 = inPtr[(49u*nThreadsPerGrid)];
            val3 = inPtr[(50u*nThreadsPerGrid)];
            val4 = inPtr[(51u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [52..55]
            //-----

            val1 = inPtr[(52u*nThreadsPerGrid)];
            val2 = inPtr[(53u*nThreadsPerGrid)];
            val3 = inPtr[(54u*nThreadsPerGrid)];
            val4 = inPtr[(55u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [56..59]
            //-----

            val1 = inPtr[(56u*nThreadsPerGrid)];
            val2 = inPtr[(57u*nThreadsPerGrid)];
            val3 = inPtr[(58u*nThreadsPerGrid)];
            val4 = inPtr[(59u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [60..63]
            //-----

            val1 = inPtr[(60u*nThreadsPerGrid)];
            val2 = inPtr[(61u*nThreadsPerGrid)];
            val3 = inPtr[(62u*nThreadsPerGrid)];
            val4 = inPtr[(63u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (64u * nThreadsPerGrid);
            nLeftOverElems -= (64u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 32
      //------

      if (K_length >= 32u)
      {
         // Process 32 chunks safely without range checking
         if (nLeftOverElems >= (32u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [9..12]
            //-----

            val1 = inPtr[( 8u*nThreadsPerGrid)];
            val2 = inPtr[( 9u*nThreadsPerGrid)];
            val3 = inPtr[(10u*nThreadsPerGrid)];
            val4 = inPtr[(11u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [13..16]
            //-----

            val1 = inPtr[(12u*nThreadsPerGrid)];
            val2 = inPtr[(13u*nThreadsPerGrid)];
            val3 = inPtr[(14u*nThreadsPerGrid)];
            val4 = inPtr[(15u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [17..20]
            //-----

            val1 = inPtr[(16u*nThreadsPerGrid)];
            val2 = inPtr[(17u*nThreadsPerGrid)];
            val3 = inPtr[(18u*nThreadsPerGrid)];
            val4 = inPtr[(19u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [21..24]
            //-----

            val1 = inPtr[(20u*nThreadsPerGrid)];
            val2 = inPtr[(21u*nThreadsPerGrid)];
            val3 = inPtr[(22u*nThreadsPerGrid)];
            val4 = inPtr[(23u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [25..28]
            //-----

            val1 = inPtr[(24u*nThreadsPerGrid)];
            val2 = inPtr[(25u*nThreadsPerGrid)];
            val3 = inPtr[(26u*nThreadsPerGrid)];
            val4 = inPtr[(27u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [29..32]
            //-----

            val1 = inPtr[(28u*nThreadsPerGrid)];
            val2 = inPtr[(29u*nThreadsPerGrid)];
            val3 = inPtr[(30u*nThreadsPerGrid)];
            val4 = inPtr[(31u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (32u * nThreadsPerGrid);
            nLeftOverElems -= (32u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 16
      //------

      if (K_length >= 16u)
      {
         // Process 16 chunks safely without range checking
         if (nLeftOverElems >= (16u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [9..12]
            //-----

            val1 = inPtr[( 8u*nThreadsPerGrid)];
            val2 = inPtr[( 9u*nThreadsPerGrid)];
            val3 = inPtr[(10u*nThreadsPerGrid)];
            val4 = inPtr[(11u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [13..16]
            //-----

            val1 = inPtr[(12u*nThreadsPerGrid)];
            val2 = inPtr[(13u*nThreadsPerGrid)];
            val3 = inPtr[(14u*nThreadsPerGrid)];
            val4 = inPtr[(15u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (16u * nThreadsPerGrid);
            nLeftOverElems -= (16u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 8
      //------

      if (K_length >= 8u)
      {
         // Process 8 chunks safely without range checking
         if (nLeftOverElems >= (8u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (8u * nThreadsPerGrid);
            nLeftOverElems -= (8u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 4
      //------

      if (K_length >= 4u)
      {
         // Process 4 chunks safely without range checking
         if (nLeftOverElems >= (4u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 

            // Process v3, v4
            Extractor::Extract4( b1, b2, b3, b4, val3, val4 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (4u * nThreadsPerGrid);
            nLeftOverElems -= (4u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 2
      //------

      if (K_length >= 2u)
      {
         // Process 2 chunks safely without range checking
         if (nLeftOverElems >= (2u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..2]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];

            // Process v1, v2
            Extractor::Extract4( b1, b2, b3, b4, val1, val2 );
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (2u * nThreadsPerGrid);
            nLeftOverElems -= (2u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 1
      //------

      if (K_length >= 1u)
      {
         // Process 1 chunk safely without range checking
         if (nLeftOverElems >= (1u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];

            // Process v1
            Extractor::Extract2( b1, b2, val1 );
            mapper.Transform2( bin1, bin2, b1, b2 );
            BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 

            // Move to next section
            baseIdx        += (1u * nThreadsPerGrid);
            nLeftOverElems -= (1u * nThreadsPerGrid);
         }
      }


      //------
      // Process Last few elements
      //    with careful RANGE CHECKING !!!
      //------

      if (nLeftOverElems > 0u)
      {
         // Make sure we are 'in range' before reading & binning
         U32 inRange1 = (baseIdx <= stop);
         if (inRange1) 
         { 
            // Read in 32-bit element
            val1 = inVals[baseIdx];

            // Process element
            Extractor::Extract2( b1, b2, val1 );
            mapper.Transform2( bin1, bin2, b1, b2 );
            BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
      }

      // Update Accumulation count
      overflow += K2_length;   // overflow += K elements * 2 words per value
	}


	// Cleanup Mapping object 
	// (Give mapper a chance to cleanup any resources)
	mapper.Finish();


   //-----
   // Accumulate 'thread' counts into 'row' counts
   //    Note: Also zeros out 'per thread' count array
   //-----

   if (overflow > 0u)
   {
      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }

      //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
      AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
      overflow = 0u;

      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }
   }



	//-------------------------------------------------
	// Write out final row 'counts'
	//-------------------------------------------------

   {
      // Compute starting 'row counts' offset
      U32 rIdx = threadIdx.x * 4u;         // 4 groups per lane
      U32 rRow = rIdx >> logBankSize;
      U32 rCol = rIdx & BankMask;

      U32 rowIdx = (rRow * strideBank) + (rCol + 1u);
         // Extra '+1' to shift past initial pad element      

      U32 * rowPtr = &s_thrdCounts[rowIdx];

      // Store row counts in row array
      rowPtr[0] = rowCnt1;
      rowPtr[1] = rowCnt2;
      rowPtr[2] = rowCnt3;
      rowPtr[3] = rowCnt4;

      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }

      // Get Warp Row & Column
      U32 warpRow = threadIdx.x >> logWarpSize; // tid / 32
      U32 warpCol = threadIdx.x & WarpMask;     // tid % 32

      // Get local & global indices
      U32 outGlobal = (blockIdx.x * nHistBins);
      U32 outLocal  = (warpRow * OutWarpSize);
      U32 rowBase   = (warpRow * OutStrideSize);
      U32 outBase   = outGlobal + outLocal;
      U32 rowOff    = warpCol + 1u;

      U32 outIdx = outBase + warpCol;
          rowIdx = rowBase + rowOff;

      // Get local & global pointers
      U32 * outPtr = &outRowCounts[outIdx];
            rowPtr = &s_thrdCounts[rowIdx];

         // Write our 'per row' counts in warp sequential order
      if (OutLength >= 1u) { outPtr[(0u*WarpSize)] = rowPtr[(0u*strideBank)]; }
      if (OutLength >= 2u) { outPtr[(1u*WarpSize)] = rowPtr[(1u*strideBank)]; }
      if (OutLength >= 3u) { outPtr[(2u*WarpSize)] = rowPtr[(2u*strideBank)]; }
      if (OutLength >= 4u) { outPtr[(3u*WarpSize)] = rowPtr[(3u*strideBank)]; }
      if (OutLength >= 5u) { outPtr[(4u*WarpSize)] = rowPtr[(4u*strideBank)]; }
      if (OutLength >= 6u) { outPtr[(5u*WarpSize)] = rowPtr[(5u*strideBank)]; }
      if (OutLength >= 7u) { outPtr[(6u*WarpSize)] = rowPtr[(6u*strideBank)]; }
      if (OutLength >= 8u) { outPtr[(7u*WarpSize)] = rowPtr[(7u*strideBank)]; }
   }
}


/*---------------------------------------------------------
  Name:   K1_TRISH_CountRows_GEN_B4
  Desc:   
  Note:   
  
  1. Assumes underlying data is stored as 32-bit values 
    (U32,I32,F32) per 32-bit storage element.
  2. This further implies that K = [1,255] to safely
     avoid overflowing an 8-bit counter.
  3. However, K >= 104 impacts performance negatively
     as the program appears to grow to be too large 
     to fit into the hardware code cache ..., 
     so we restrict K to the range K=[1..127]
 ---------------------------------------------------------*/

template < 
           typename valT,	   // underlying value Type (U8, I8)
           typename mapT,	   // underlying mapper object 
           U32 logBankSize,	// log<2>( Channels per Bank )
           U32 logWarpSize,	// log<2>( Threads per Warp )
           U32 BlockSize,	   // Threads Per Block (needs to be a power of 2 & multiple of warpsize)
		     U32 GridSize,	   // Blocks Per Grid
           U32 K_length 	   // #elements to process per thread before looping
         >
__global__
void K1_TRISH_CountRows_GEN_B4
( 
         U32  * outRowCounts,	// OUT - 256-way row-sums array
   const valT * inVals,			// IN  - values to bin and count
         U32    start,			// IN  - range [start,stop] to check and count
         U32    stop,			//       ditto
         valT   minVal,			// IN  - minimum value
         valT   maxVal,			// IN  - maximum value
         U32    numBins       // IN  - number of bins (in histogram)
) 
{
   //-------------------------------------------
   // Constant values (computed at compile time)
   //-------------------------------------------

	   // Bank Size (elements per bank)
   const U32 BankSize    = (1u << logBankSize);	   // 32 = 2^5 threads per bank
   const U32 BankMask    = BankSize - 1u;	         // 31 = 32 - 1 = 0x1F = b11111
   const U32 strideBank  = BankSize + 1u;          // 33 = 32 + 1
      // Extra '+1' to help try and avoid bank conflicts

	   // Warp Size (threads per warp)
   const U32 WarpSize    = (1u << logWarpSize);	   // 32 = 2^5 threads per warp
   const U32 WarpMask    = WarpSize - 1u;			   // 31 = 32 - 1 = 0x1F = b11111

      // Block Size (threads per block)
   //const U32 BlockSize   = 64u;
   const U32 BlockMask   = BlockSize - 1u;

	   // Chunk Size
   //const U32 ChunkSize     = BlockSize * K_length;
   //const U32 IN_WarpSize   = K_length * WarpSize;

      // K_length
   //const U32 K_length = 16u;               //  16 
   const U32 K1_length = K_length;           //  16 = 16  (1 storage value per input value)
   const U32 K1_stop   = 256u - K1_length;   // 240 = 256 - 16 (conservative test)

	   // Warps Per Block
   const U32 WarpsPerBlock = BlockSize / WarpSize;   // 2 = 64/32

	   // Bins per Histogram
   const U32 nHistBins     = 256u;     // 256 = 2^8

	   // Lane Info (Compress 4 'bins' into each 32-bit value)
   const U32 nLanes		   = 64u;   // 64, # Lanes = 256 bins / 4 bins per lane

	   // 'Per Thread' counts array
   const U32 nTCounts      = nLanes * BlockSize;
   const U32 banksTCounts  = (nTCounts + BankMask) / BankSize;
   const U32 padTCounts    = (banksTCounts * BankSize) - nTCounts;
   const U32 sizeTCounts   = nTCounts + padTCounts;

      // Output size
   const U32 OutWarpSize   = nHistBins / WarpsPerBlock;
   const U32 OutLength     = OutWarpSize / WarpSize;
   const U32 OutStrideSize = OutLength * strideBank;

	   // Array Initialization
   const U32 nPassesThrd  = sizeTCounts / BlockSize;
   const U32 leftOverThrd = sizeTCounts - (nPassesThrd * BlockSize);

   const U32 nThreadsPerGrid = BlockSize * GridSize;	//   3,072 = 64 * 48
   const U32 rowSize = K_length * nThreadsPerGrid;		// 193,586 = 63 * 64 * 48


   //------------------------------------
   // Local Typedefs
   //------------------------------------

   // TRISH types
   typedef typename TRISH_traits<valT>::base_type    baseType;
   typedef typename TRISH_traits<valT>::bin_type     binType;
   typedef typename TRISH_traits<valT>::upscale_type upscaleType;
   typedef typename TRISH_traits<valT>::convert_type convertType;


   //------------------------------------
   // Local Variables
   //------------------------------------

	   // Local variables (shared memory)
   __shared__ U32  s_thrdCounts[sizeTCounts];   // 'per thread' counts

      // Local variables (registers)
   U32 rowCnt1 = 0u;
   U32 rowCnt2 = 0u;
   U32 rowCnt3 = 0u; 
   U32 rowCnt4 = 0u;


   //---------------------------
   // Compute Indices & Pointers
   //---------------------------

   U32 tid = threadIdx.x;		// Thread ID within Block
   U32 * cntPtr;
   U32 * basePtr;

   {
      // Get Warp Row & Column
      //U32 warpRow = threadIdx.x >> logWarpSize; // tid / 32
      //U32 warpCol = threadIdx.x & WarpMask;     // tid % 32

      // Compute starting 'input' offset (Warp Sequential Layout)
      //inIdx = (warpRow * IN_WarpSize) // Move to each warps assigned portion of work
      //        + warpCol;              // Move to warp column (in warp)

         // Compute starting serial scan index
      U32 baseIdx = (tid * BlockSize);

         // Get pointers into shared memory array
         // for different views of memory
      cntPtr  = &s_thrdCounts[threadIdx.x];
      basePtr = &s_thrdCounts[baseIdx];
   }


   //-------------------------------------------
   // Zero out arrays
   //-------------------------------------------

   {
	   //-
	   // Zero out 'Per Thread' counts
	   //-

      U32 * ptrTC = (&s_thrdCounts[0]);
      SetArray_BlockSeq
         < 
            U32, BlockSize, nPassesThrd, leftOverThrd, sizeTCounts
         >
         ( 
            ptrTC, 0u
         );
   }


   //-----
   // Compute thread, block, & grid indices & sizes
   //-----
 
   U32 bid = (blockIdx.y * gridDim.x) + blockIdx.x;		// Block ID within Grid
   U32 elemOffset = (bid * K_length * BlockSize) + tid;	// Starting offset 

   U32 nElems32        = stop - start + 1u;
   U32 nMaxRows        = (nElems32 + (rowSize - 1u)) / rowSize;
   U32 nSafeRows       = nElems32 / rowSize;
   U32 nSafeElems      = nSafeRows * rowSize;
   U32 nLeftOverElems  = nElems32 - nSafeElems;

   U32 startIdx        = start + elemOffset;
   U32 stopIdx         = startIdx + (nSafeRows * rowSize);
   U32 currIdx         = startIdx;
   U32 overflow        = 0u;

   // Initiate 

   // Initiate Mapping object 
   // (Transform from values to bin indices)
   mapT mapper;
   mapper.Initiate( minVal, maxVal, numBins );


   //-----
   // Process all safe blocks
   //-----

   // 'input' pointer for reading from memory
   const valT * inPtr = &inVals[currIdx];

   // Sync Threads in Block
   if (WarpsPerBlock >= 2u) { __syncthreads(); }

   while (currIdx < stopIdx)
   {
      //-----
      // Accumulate 'thread' counts into 'row' counts
      //    Note: Also zeros out 'per thread' count array
      //-----

      if (overflow >= K1_stop)
      {
         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }

         //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         overflow = 0u;

         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }
      }

      valT        val1, val2, val3, val4;
      upscaleType b1, b2, b3, b4;
      binType     bin1, bin2, bin3, bin4;

         // NOTE:  the 'K_length' variable below is a static
         //        hard-coded constant in the range [1..63].
         //        K = 'Work per thread' per loop (stride)...
         //        The compiler will take care of throwing away 
         //        any unused code greater than our specified 'K'
         //        value, with no negative impact on performance.

      //-
      // Process values [0..3]
      //-

      // Read in first 'four' values (32-bit)
      if (K_length >= 1u) { val1 = inPtr[0u*BlockSize]; }
      if (K_length >= 2u) { val2 = inPtr[1u*BlockSize]; }
      if (K_length >= 3u) { val3 = inPtr[2u*BlockSize]; }
      if (K_length >= 4u) { val4 = inPtr[3u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 1u) { b1 = upscaleType(val1); }
      if (K_length >= 2u) { b2 = upscaleType(val2); }
      if (K_length >= 3u) { b3 = upscaleType(val3); }
      if (K_length >= 4u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 4u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 3u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 2u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 1u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [4..7]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 5u) { val1 = inPtr[4u*BlockSize]; }
      if (K_length >= 6u) { val2 = inPtr[5u*BlockSize]; }
      if (K_length >= 7u) { val3 = inPtr[6u*BlockSize]; }
      if (K_length >= 8u) { val4 = inPtr[7u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 5u) { b1 = upscaleType(val1); }
      if (K_length >= 6u) { b2 = upscaleType(val2); }
      if (K_length >= 7u) { b3 = upscaleType(val3); }
      if (K_length >= 8u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 8u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 7u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 6u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 5u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [8..11]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >=  9u) { val1 = inPtr[ 8u*BlockSize]; }
      if (K_length >= 10u) { val2 = inPtr[ 9u*BlockSize]; }
      if (K_length >= 11u) { val3 = inPtr[10u*BlockSize]; }
      if (K_length >= 12u) { val4 = inPtr[11u*BlockSize]; }

      // Convert to upscale type
      if (K_length >=  9u) { b1 = upscaleType(val1); }
      if (K_length >= 10u) { b2 = upscaleType(val2); }
      if (K_length >= 11u) { b3 = upscaleType(val3); }
      if (K_length >= 12u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 12u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 11u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 10u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length ==  9u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [12..15]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 13u) { val1 = inPtr[12u*BlockSize]; }
      if (K_length >= 14u) { val2 = inPtr[13u*BlockSize]; }
      if (K_length >= 15u) { val3 = inPtr[14u*BlockSize]; }
      if (K_length >= 16u) { val4 = inPtr[15u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 13u) { b1 = upscaleType(val1); }
      if (K_length >= 14u) { b2 = upscaleType(val2); }
      if (K_length >= 15u) { b3 = upscaleType(val3); }
      if (K_length >= 16u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 16u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 15u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 14u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 13u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [16..19]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 17u) { val1 = inPtr[16u*BlockSize]; }
      if (K_length >= 18u) { val2 = inPtr[17u*BlockSize]; }
      if (K_length >= 19u) { val3 = inPtr[18u*BlockSize]; }
      if (K_length >= 20u) { val4 = inPtr[19u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 17u) { b1 = upscaleType(val1); }
      if (K_length >= 18u) { b2 = upscaleType(val2); }
      if (K_length >= 19u) { b3 = upscaleType(val3); }
      if (K_length >= 20u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 20u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 19u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 18u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 17u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [20..23]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 21u) { val1 = inPtr[20u*BlockSize]; }
      if (K_length >= 22u) { val2 = inPtr[21u*BlockSize]; }
      if (K_length >= 23u) { val3 = inPtr[22u*BlockSize]; }
      if (K_length >= 24u) { val4 = inPtr[23u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 21u) { b1 = upscaleType(val1); }
      if (K_length >= 22u) { b2 = upscaleType(val2); }
      if (K_length >= 23u) { b3 = upscaleType(val3); }
      if (K_length >= 24u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 24u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 23u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 22u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 21u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [24..27]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 25u) { val1 = inPtr[24u*BlockSize]; }
      if (K_length >= 26u) { val2 = inPtr[25u*BlockSize]; }
      if (K_length >= 27u) { val3 = inPtr[26u*BlockSize]; }
      if (K_length >= 28u) { val4 = inPtr[27u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 25u) { b1 = upscaleType(val1); }
      if (K_length >= 26u) { b2 = upscaleType(val2); }
      if (K_length >= 27u) { b3 = upscaleType(val3); }
      if (K_length >= 28u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 28u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 27u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 26u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 25u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [28..31]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 29u) { val1 = inPtr[28u*BlockSize]; }
      if (K_length >= 30u) { val2 = inPtr[29u*BlockSize]; }
      if (K_length >= 31u) { val3 = inPtr[30u*BlockSize]; }
      if (K_length >= 32u) { val4 = inPtr[31u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 29u) { b1 = upscaleType(val1); }
      if (K_length >= 30u) { b2 = upscaleType(val2); }
      if (K_length >= 31u) { b3 = upscaleType(val3); }
      if (K_length >= 32u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 32u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 31u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 30u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 29u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [32..35]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 33u) { val1 = inPtr[32u*BlockSize]; }
      if (K_length >= 34u) { val2 = inPtr[33u*BlockSize]; }
      if (K_length >= 35u) { val3 = inPtr[34u*BlockSize]; }
      if (K_length >= 36u) { val4 = inPtr[35u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 33u) { b1 = upscaleType(val1); }
      if (K_length >= 34u) { b2 = upscaleType(val2); }
      if (K_length >= 35u) { b3 = upscaleType(val3); }
      if (K_length >= 36u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 36u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 35u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 34u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 33u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [36..39]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 37u) { val1 = inPtr[36u*BlockSize]; }
      if (K_length >= 38u) { val2 = inPtr[37u*BlockSize]; }
      if (K_length >= 39u) { val3 = inPtr[38u*BlockSize]; }
      if (K_length >= 40u) { val4 = inPtr[39u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 37u) { b1 = upscaleType(val1); }
      if (K_length >= 38u) { b2 = upscaleType(val2); }
      if (K_length >= 39u) { b3 = upscaleType(val3); }
      if (K_length >= 40u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 40u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 39u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 38u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 37u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [40..43]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 41u) { val1 = inPtr[40u*BlockSize]; }
      if (K_length >= 42u) { val2 = inPtr[41u*BlockSize]; }
      if (K_length >= 43u) { val3 = inPtr[42u*BlockSize]; }
      if (K_length >= 44u) { val4 = inPtr[43u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 41u) { b1 = upscaleType(val1); }
      if (K_length >= 42u) { b2 = upscaleType(val2); }
      if (K_length >= 43u) { b3 = upscaleType(val3); }
      if (K_length >= 44u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 44u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 43u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 42u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 41u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [44..47]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 45u) { val1 = inPtr[44u*BlockSize]; }
      if (K_length >= 46u) { val2 = inPtr[45u*BlockSize]; }
      if (K_length >= 47u) { val3 = inPtr[46u*BlockSize]; }
      if (K_length >= 48u) { val4 = inPtr[47u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 45u) { b1 = upscaleType(val1); }
      if (K_length >= 46u) { b2 = upscaleType(val2); }
      if (K_length >= 47u) { b3 = upscaleType(val3); }
      if (K_length >= 48u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 48u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 47u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 46u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 45u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [48..51]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 49u) { val1 = inPtr[48u*BlockSize]; }
      if (K_length >= 50u) { val2 = inPtr[49u*BlockSize]; }
      if (K_length >= 51u) { val3 = inPtr[50u*BlockSize]; }
      if (K_length >= 52u) { val4 = inPtr[51u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 49u) { b1 = upscaleType(val1); }
      if (K_length >= 50u) { b2 = upscaleType(val2); }
      if (K_length >= 51u) { b3 = upscaleType(val3); }
      if (K_length >= 52u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 52u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 51u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 50u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 49u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [52..55]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 53u) { val1 = inPtr[52u*BlockSize]; }
      if (K_length >= 54u) { val2 = inPtr[53u*BlockSize]; }
      if (K_length >= 55u) { val3 = inPtr[54u*BlockSize]; }
      if (K_length >= 56u) { val4 = inPtr[55u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 53u) { b1 = upscaleType(val1); }
      if (K_length >= 54u) { b2 = upscaleType(val2); }
      if (K_length >= 55u) { b3 = upscaleType(val3); }
      if (K_length >= 56u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 56u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 55u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 54u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 53u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [56..59]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 57u) { val1 = inPtr[56u*BlockSize]; }
      if (K_length >= 58u) { val2 = inPtr[57u*BlockSize]; }
      if (K_length >= 59u) { val3 = inPtr[58u*BlockSize]; }
      if (K_length >= 60u) { val4 = inPtr[59u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 57u) { b1 = upscaleType(val1); }
      if (K_length >= 58u) { b2 = upscaleType(val2); }
      if (K_length >= 59u) { b3 = upscaleType(val3); }
      if (K_length >= 60u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 60u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 59u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 58u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 57u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [60..63]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 61u) { val1 = inPtr[60u*BlockSize]; }
      if (K_length >= 62u) { val2 = inPtr[61u*BlockSize]; }
      if (K_length >= 63u) { val3 = inPtr[62u*BlockSize]; }
      if (K_length >= 64u) { val4 = inPtr[63u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 61u) { b1 = upscaleType(val1); }
      if (K_length >= 62u) { b2 = upscaleType(val2); }
      if (K_length >= 63u) { b3 = upscaleType(val3); }
      if (K_length >= 64u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 64u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 63u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 62u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 61u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [64..67]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 65u) { val1 = inPtr[64u*BlockSize]; }
      if (K_length >= 66u) { val2 = inPtr[65u*BlockSize]; }
      if (K_length >= 67u) { val3 = inPtr[66u*BlockSize]; }
      if (K_length >= 68u) { val4 = inPtr[67u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 65u) { b1 = upscaleType(val1); }
      if (K_length >= 66u) { b2 = upscaleType(val2); }
      if (K_length >= 67u) { b3 = upscaleType(val3); }
      if (K_length >= 68u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 68u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 67u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 66u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 65u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [68..71]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 69u) { val1 = inPtr[68u*BlockSize]; }
      if (K_length >= 70u) { val2 = inPtr[69u*BlockSize]; }
      if (K_length >= 71u) { val3 = inPtr[70u*BlockSize]; }
      if (K_length >= 72u) { val4 = inPtr[71u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 69u) { b1 = upscaleType(val1); }
      if (K_length >= 70u) { b2 = upscaleType(val2); }
      if (K_length >= 71u) { b3 = upscaleType(val3); }
      if (K_length >= 72u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 72u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 71u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 70u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 69u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [72..75]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 73u) { val1 = inPtr[72u*BlockSize]; }
      if (K_length >= 74u) { val2 = inPtr[73u*BlockSize]; }
      if (K_length >= 75u) { val3 = inPtr[74u*BlockSize]; }
      if (K_length >= 76u) { val4 = inPtr[75u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 73u) { b1 = upscaleType(val1); }
      if (K_length >= 74u) { b2 = upscaleType(val2); }
      if (K_length >= 75u) { b3 = upscaleType(val3); }
      if (K_length >= 76u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 76u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 75u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 74u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 73u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [76..79]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 77u) { val1 = inPtr[76u*BlockSize]; }
      if (K_length >= 78u) { val2 = inPtr[77u*BlockSize]; }
      if (K_length >= 79u) { val3 = inPtr[78u*BlockSize]; }
      if (K_length >= 80u) { val4 = inPtr[79u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 77u) { b1 = upscaleType(val1); }
      if (K_length >= 78u) { b2 = upscaleType(val2); }
      if (K_length >= 79u) { b3 = upscaleType(val3); }
      if (K_length >= 80u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 80u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 79u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 78u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 77u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [80..83]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 81u) { val1 = inPtr[80u*BlockSize]; }
      if (K_length >= 82u) { val2 = inPtr[81u*BlockSize]; }
      if (K_length >= 83u) { val3 = inPtr[82u*BlockSize]; }
      if (K_length >= 84u) { val4 = inPtr[83u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 81u) { b1 = upscaleType(val1); }
      if (K_length >= 82u) { b2 = upscaleType(val2); }
      if (K_length >= 83u) { b3 = upscaleType(val3); }
      if (K_length >= 84u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 84u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 83u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 82u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 81u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [84..87]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 85u) { val1 = inPtr[84u*BlockSize]; }
      if (K_length >= 86u) { val2 = inPtr[85u*BlockSize]; }
      if (K_length >= 87u) { val3 = inPtr[86u*BlockSize]; }
      if (K_length >= 88u) { val4 = inPtr[87u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 85u) { b1 = upscaleType(val1); }
      if (K_length >= 86u) { b2 = upscaleType(val2); }
      if (K_length >= 87u) { b3 = upscaleType(val3); }
      if (K_length >= 88u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 88u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 87u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 86u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 85u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [88..91]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 89u) { val1 = inPtr[88u*BlockSize]; }
      if (K_length >= 90u) { val2 = inPtr[89u*BlockSize]; }
      if (K_length >= 91u) { val3 = inPtr[90u*BlockSize]; }
      if (K_length >= 92u) { val4 = inPtr[91u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 89u) { b1 = upscaleType(val1); }
      if (K_length >= 90u) { b2 = upscaleType(val2); }
      if (K_length >= 91u) { b3 = upscaleType(val3); }
      if (K_length >= 92u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 92u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 91u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 90u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 89u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [92..95]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 93u) { val1 = inPtr[92u*BlockSize]; }
      if (K_length >= 94u) { val2 = inPtr[93u*BlockSize]; }
      if (K_length >= 95u) { val3 = inPtr[94u*BlockSize]; }
      if (K_length >= 96u) { val4 = inPtr[95u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 93u) { b1 = upscaleType(val1); }
      if (K_length >= 94u) { b2 = upscaleType(val2); }
      if (K_length >= 95u) { b3 = upscaleType(val3); }
      if (K_length >= 96u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 96u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 95u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 94u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 93u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [96..99]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >=  97u) { val1 = inPtr[96u*BlockSize]; }
      if (K_length >=  98u) { val2 = inPtr[97u*BlockSize]; }
      if (K_length >=  99u) { val3 = inPtr[98u*BlockSize]; }
      if (K_length >= 100u) { val4 = inPtr[99u*BlockSize]; }

      // Convert to upscale type
      if (K_length >=  97u) { b1 = upscaleType(val1); }
      if (K_length >=  98u) { b2 = upscaleType(val2); }
      if (K_length >=  99u) { b3 = upscaleType(val3); }
      if (K_length >= 100u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 100u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length ==  99u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length ==  98u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length ==  97u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [100..103]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 101u) { val1 = inPtr[100u*BlockSize]; }
      if (K_length >= 102u) { val2 = inPtr[101u*BlockSize]; }
      if (K_length >= 103u) { val3 = inPtr[102u*BlockSize]; }
      if (K_length >= 104u) { val4 = inPtr[103u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 101u) { b1 = upscaleType(val1); }
      if (K_length >= 102u) { b2 = upscaleType(val2); }
      if (K_length >= 103u) { b3 = upscaleType(val3); }
      if (K_length >= 104u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 104u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 103u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 102u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 101u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [104..107]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 105u) { val1 = inPtr[104u*BlockSize]; }
      if (K_length >= 106u) { val2 = inPtr[105u*BlockSize]; }
      if (K_length >= 107u) { val3 = inPtr[106u*BlockSize]; }
      if (K_length >= 108u) { val4 = inPtr[107u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 105u) { b1 = upscaleType(val1); }
      if (K_length >= 106u) { b2 = upscaleType(val2); }
      if (K_length >= 107u) { b3 = upscaleType(val3); }
      if (K_length >= 108u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 108u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 107u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 106u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 105u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [108..111]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 109u) { val1 = inPtr[108u*BlockSize]; }
      if (K_length >= 110u) { val2 = inPtr[109u*BlockSize]; }
      if (K_length >= 111u) { val3 = inPtr[110u*BlockSize]; }
      if (K_length >= 112u) { val4 = inPtr[111u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 109u) { b1 = upscaleType(val1); }
      if (K_length >= 110u) { b2 = upscaleType(val2); }
      if (K_length >= 111u) { b3 = upscaleType(val3); }
      if (K_length >= 112u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 112u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 111u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 110u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 109u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [112..115]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 113u) { val1 = inPtr[112u*BlockSize]; }
      if (K_length >= 114u) { val2 = inPtr[113u*BlockSize]; }
      if (K_length >= 115u) { val3 = inPtr[114u*BlockSize]; }
      if (K_length >= 116u) { val4 = inPtr[115u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 113u) { b1 = upscaleType(val1); }
      if (K_length >= 114u) { b2 = upscaleType(val2); }
      if (K_length >= 115u) { b3 = upscaleType(val3); }
      if (K_length >= 116u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 116u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 115u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 114u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 113u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [116..119]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 117u) { val1 = inPtr[116u*BlockSize]; }
      if (K_length >= 118u) { val2 = inPtr[117u*BlockSize]; }
      if (K_length >= 119u) { val3 = inPtr[118u*BlockSize]; }
      if (K_length >= 120u) { val4 = inPtr[119u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 117u) { b1 = upscaleType(val1); }
      if (K_length >= 118u) { b2 = upscaleType(val2); }
      if (K_length >= 119u) { b3 = upscaleType(val3); }
      if (K_length >= 120u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 120u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 119u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 118u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 117u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [120..123]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 121u) { val1 = inPtr[120u*BlockSize]; }
      if (K_length >= 122u) { val2 = inPtr[121u*BlockSize]; }
      if (K_length >= 123u) { val3 = inPtr[122u*BlockSize]; }
      if (K_length >= 124u) { val4 = inPtr[123u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 121u) { b1 = upscaleType(val1); }
      if (K_length >= 122u) { b2 = upscaleType(val2); }
      if (K_length >= 123u) { b3 = upscaleType(val3); }
      if (K_length >= 124u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 124u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 123u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 122u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 121u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }


      //-
      // Process values [124..127]
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 125u) { val1 = inPtr[124u*BlockSize]; }
      if (K_length >= 126u) { val2 = inPtr[125u*BlockSize]; }
      if (K_length >= 127u) { val3 = inPtr[126u*BlockSize]; }
      if (K_length >= 128u) { val4 = inPtr[127u*BlockSize]; }

      // Convert to upscale type
      if (K_length >= 125u) { b1 = upscaleType(val1); }
      if (K_length >= 126u) { b2 = upscaleType(val2); }
      if (K_length >= 127u) { b3 = upscaleType(val3); }
      if (K_length >= 128u) { b4 = upscaleType(val4); }

      // Bin first 'four' values into count array
      if (K_length >= 128u) 
      { 
         // Process v1,v2,v3,v4
	      mapper.Transform4( bin1, bin2, bin3, bin4, // OUT => bins
                            b1,   b2,   b3,   b4 ); // IN => values to transform
	      BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 
      }
      else 
      {
         if (K_length == 127u)
         {
            // Process v1,v2,v3
	         mapper.Transform3( bin1, bin2, bin3, // OUT => bins
                               b1,   b2,   b3 ); // IN => values to transform
	         BinCount3<BlockSize>( cntPtr, bin1, bin2, bin3 ); 
         }
         if (K_length == 126u)
         {
            // Process v1,v2
	         mapper.Transform2( bin1, bin2, // OUT => bins
                               b1,   b2 ); // IN => values to transform
	         BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 
         }
         if (K_length == 125u)
         {
            // Process v1
	         mapper.Transform1( bin1, b1 );
	         BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }

      //
      // Note:  We could repeat the above pattern all the way up to 
      //        K = [252..255] making sure to deliberately skip
      //        K = 255 (the 256th value) to avoid overflow
      //  However, somewhere around K = 104, we appear to overflow
      //  the hardware code cache anyway which negatively impacts
      //  performance, so we don't need to go all the way...
      //


      //-----
      // Move to next row of work
      //-----

      currIdx += rowSize;
        inPtr += rowSize;

      // Increment 'overflow' count
      overflow += K1_length;   // K values
	}

   __syncthreads();


   //--------------------------------------
   // LAST: Process last leftover chunk
   //       with more careful range checking
   //--------------------------------------

   if (nLeftOverElems)
   {
      //-----
      // Accumulate 'thread' counts into 'row' counts
      //    Note: Also zeros out 'per thread' count array
      //-----

      if (overflow >= K1_stop)
      {
         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }

         //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         overflow = 0u;

         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }
      }

         // NOTE #1:  the 'K_length' variable below is a static
         //           hard-coded constant in the range [1..255].
         //           K = 'Work per thread' per loop (stride)...
         //           The compiler will take care of throwing away 
         //           any unused code greater than our specified 'K'
         //           value, with no negative impact on performance.

         // NOTE #2:  We use a cooperative stride 
         //           across each thread in each block in grid
         //           ChunkSize = BlockSize * GridSize = 64 * 48 = 3072
         //           RowSize   = WorkPerThead(K) * ChunkSize = 63 * 3072 = 193,536
         // 
         //                       B0   B1  ...  B47  (Blocks in Grid)
         //                      ---- ---- --- ----
         //           k =  1 =>  |64| |64| ... |64|  (3072 Thread & I/O requests for 1st work item per thread)
         //           k =  2 =>  |64| |64| ... |64|  ditto (2nd work item per thread)
         //               ...       ...         ...
         //           k = 63 =>  |64| |64| ... |64|  ditto (63 work item per thread)

         // NOTE #3:  We use a "Divide & Conquer" approach 
         //           to avoid as much slower range checking as possible
         //			  We try batches of 128, 64, 32, 16, 8, 4, 2, 1, 
         //         and then finally a leftover chunk (on which we must carefully range check) 

      //----
      // Setup Pointers & Indices for cooperative stride 
      //----

      U32 bid        = (blockIdx.y * gridDim.x) + blockIdx.x;	// Get block index
      U32 nSkip      = nSafeRows * rowSize;						   // Skip past already processed rows
      U32 chunkIdx   = (bid * BlockSize) + tid;					   // Get starting index within chunk
      U32 baseIdx    = start + nSkip + chunkIdx;				   // Get starting index for left over elements

      U32         val1, val2, val3, val4;
      upscaleType b1, b2, b3, b4;
      binType     bin1, bin2, bin3, bin4;

      //------
      // Try Section of 128
      //------

      //
      // Note: We didn't bother to insert this code due to the "code cache" performance problem
      //       for K >= 104.
      //
      //       If desired, repeat the pattern for the section of 64 below
      //       while doubling the # of elements processed.


      //------
      // Try Section of 64
      //------

      if (K_length >= 64u)
      {
         // Process 64 chunks safely without range checking
         if (nLeftOverElems >= (64u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [9..12]
            //-----

            val1 = inPtr[( 8u*nThreadsPerGrid)];
            val2 = inPtr[( 9u*nThreadsPerGrid)];
            val3 = inPtr[(10u*nThreadsPerGrid)];
            val4 = inPtr[(11u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [13..16]
            //-----

            val1 = inPtr[(12u*nThreadsPerGrid)];
            val2 = inPtr[(13u*nThreadsPerGrid)];
            val3 = inPtr[(14u*nThreadsPerGrid)];
            val4 = inPtr[(15u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [17..20]
            //-----

            val1 = inPtr[(16u*nThreadsPerGrid)];
            val2 = inPtr[(17u*nThreadsPerGrid)];
            val3 = inPtr[(18u*nThreadsPerGrid)];
            val4 = inPtr[(19u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [21..24]
            //-----

            val1 = inPtr[(20u*nThreadsPerGrid)];
            val2 = inPtr[(21u*nThreadsPerGrid)];
            val3 = inPtr[(22u*nThreadsPerGrid)];
            val4 = inPtr[(23u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [25..28]
            //-----

            val1 = inPtr[(24u*nThreadsPerGrid)];
            val2 = inPtr[(25u*nThreadsPerGrid)];
            val3 = inPtr[(26u*nThreadsPerGrid)];
            val4 = inPtr[(27u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [29..32]
            //-----

            val1 = inPtr[(28u*nThreadsPerGrid)];
            val2 = inPtr[(29u*nThreadsPerGrid)];
            val3 = inPtr[(30u*nThreadsPerGrid)];
            val4 = inPtr[(31u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [33..35]
            //-----

            val1 = inPtr[(32u*nThreadsPerGrid)];
            val2 = inPtr[(33u*nThreadsPerGrid)];
            val3 = inPtr[(34u*nThreadsPerGrid)];
            val4 = inPtr[(35u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [36..39]
            //-----

            val1 = inPtr[(36u*nThreadsPerGrid)];
            val2 = inPtr[(37u*nThreadsPerGrid)];
            val3 = inPtr[(38u*nThreadsPerGrid)];
            val4 = inPtr[(39u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [40..43]
            //-----

            val1 = inPtr[(40u*nThreadsPerGrid)];
            val2 = inPtr[(41u*nThreadsPerGrid)];
            val3 = inPtr[(42u*nThreadsPerGrid)];
            val4 = inPtr[(43u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [44..47]
            //-----

            val1 = inPtr[(44u*nThreadsPerGrid)];
            val2 = inPtr[(45u*nThreadsPerGrid)];
            val3 = inPtr[(46u*nThreadsPerGrid)];
            val4 = inPtr[(47u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [48..51]
            //-----

            val1 = inPtr[(48u*nThreadsPerGrid)];
            val2 = inPtr[(49u*nThreadsPerGrid)];
            val3 = inPtr[(50u*nThreadsPerGrid)];
            val4 = inPtr[(51u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [52..55]
            //-----

            val1 = inPtr[(52u*nThreadsPerGrid)];
            val2 = inPtr[(53u*nThreadsPerGrid)];
            val3 = inPtr[(54u*nThreadsPerGrid)];
            val4 = inPtr[(55u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [56..59]
            //-----

            val1 = inPtr[(56u*nThreadsPerGrid)];
            val2 = inPtr[(57u*nThreadsPerGrid)];
            val3 = inPtr[(58u*nThreadsPerGrid)];
            val4 = inPtr[(59u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [60..63]
            //-----

            val1 = inPtr[(60u*nThreadsPerGrid)];
            val2 = inPtr[(61u*nThreadsPerGrid)];
            val3 = inPtr[(62u*nThreadsPerGrid)];
            val4 = inPtr[(63u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (64u * nThreadsPerGrid);
            nLeftOverElems -= (64u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 32
      //------

      if (K_length >= 32u)
      {
         // Process 32 chunks safely without range checking
         if (nLeftOverElems >= (32u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [9..12]
            //-----

            val1 = inPtr[( 8u*nThreadsPerGrid)];
            val2 = inPtr[( 9u*nThreadsPerGrid)];
            val3 = inPtr[(10u*nThreadsPerGrid)];
            val4 = inPtr[(11u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [13..16]
            //-----

            val1 = inPtr[(12u*nThreadsPerGrid)];
            val2 = inPtr[(13u*nThreadsPerGrid)];
            val3 = inPtr[(14u*nThreadsPerGrid)];
            val4 = inPtr[(15u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [17..20]
            //-----

            val1 = inPtr[(16u*nThreadsPerGrid)];
            val2 = inPtr[(17u*nThreadsPerGrid)];
            val3 = inPtr[(18u*nThreadsPerGrid)];
            val4 = inPtr[(19u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [21..24]
            //-----

            val1 = inPtr[(20u*nThreadsPerGrid)];
            val2 = inPtr[(21u*nThreadsPerGrid)];
            val3 = inPtr[(22u*nThreadsPerGrid)];
            val4 = inPtr[(23u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [25..28]
            //-----

            val1 = inPtr[(24u*nThreadsPerGrid)];
            val2 = inPtr[(25u*nThreadsPerGrid)];
            val3 = inPtr[(26u*nThreadsPerGrid)];
            val4 = inPtr[(27u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [29..32]
            //-----

            val1 = inPtr[(28u*nThreadsPerGrid)];
            val2 = inPtr[(29u*nThreadsPerGrid)];
            val3 = inPtr[(30u*nThreadsPerGrid)];
            val4 = inPtr[(31u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (32u * nThreadsPerGrid);
            nLeftOverElems -= (32u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 16
      //------

      if (K_length >= 16u)
      {
         // Process 16 chunks safely without range checking
         if (nLeftOverElems >= (16u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [9..12]
            //-----

            val1 = inPtr[( 8u*nThreadsPerGrid)];
            val2 = inPtr[( 9u*nThreadsPerGrid)];
            val3 = inPtr[(10u*nThreadsPerGrid)];
            val4 = inPtr[(11u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [13..16]
            //-----

            val1 = inPtr[(12u*nThreadsPerGrid)];
            val2 = inPtr[(13u*nThreadsPerGrid)];
            val3 = inPtr[(14u*nThreadsPerGrid)];
            val4 = inPtr[(15u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (16u * nThreadsPerGrid);
            nLeftOverElems -= (16u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 8
      //------

      if (K_length >= 8u)
      {
         // Process 8 chunks safely without range checking
         if (nLeftOverElems >= (8u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (8u * nThreadsPerGrid);
            nLeftOverElems -= (8u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 4
      //------

      if (K_length >= 4u)
      {
         // Process 4 chunks safely without range checking
         if (nLeftOverElems >= (4u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);
            b3 = upscaleType(val3);
            b4 = upscaleType(val4);

            // Process v1, v2, v3, v4
            mapper.Transform4( bin1, bin2, bin3, bin4,
                               b1,   b2,   b3,   b4 );
            BinCount4<BlockSize>( cntPtr, bin1, bin2, bin3, bin4 ); 


            // Move to next section
            baseIdx        += (4u * nThreadsPerGrid);
            nLeftOverElems -= (4u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 2
      //------

      if (K_length >= 2u)
      {
         // Process 2 chunks safely without range checking
         if (nLeftOverElems >= (2u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..2]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);
            b2 = upscaleType(val2);

            // Process v1, v2
            mapper.Transform2( bin1, bin2, b1,   b2 );
            BinCount2<BlockSize>( cntPtr, bin1, bin2 ); 


            // Move to next section
            baseIdx        += (2u * nThreadsPerGrid);
            nLeftOverElems -= (2u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 1
      //------

      if (K_length >= 1u)
      {
         // Process 1 chunk safely without range checking
         if (nLeftOverElems >= (1u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];

            // Convert to upscale type
            b1 = upscaleType(val1);

            // Process v1
            mapper.Transform1( bin1, b1 );
            BinCount1<BlockSize>( cntPtr, bin1 ); 

            // Move to next section
            baseIdx        += (1u * nThreadsPerGrid);
            nLeftOverElems -= (1u * nThreadsPerGrid);
         }
      }


      //------
      // Process Last few elements
      //    with careful RANGE CHECKING !!!
      //------

      if (nLeftOverElems > 0u)
      {
         // Make sure we are 'in range' before reading & binning
         U32 inRange1 = (baseIdx <= stop);
         if (inRange1) 
         { 
            // Read in 32-bit element
            val1 = inVals[baseIdx];

            // Process single element
            b1 = upscaleType(val1);
            mapper.Transform1( bin1, b1 );
            BinCount1<BlockSize>( cntPtr, bin1 ); 
         }
      }

      // Update Accumulation count
      overflow += K1_length;   // overflow += K elements
	}


	// Cleanup Mapping object 
	// (Give mapper a chance to cleanup any resources)
	mapper.Finish();


   //-----
   // Accumulate 'thread' counts into 'row' counts
   //    Note: Also zeros out 'per thread' count array
   //-----

   if (overflow > 0u)
   {
      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }

      //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
      AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
      overflow = 0u;

      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }
   }



	//-------------------------------------------------
	// Write out final row 'counts'
	//-------------------------------------------------

   {
      // Compute starting 'row counts' offset
      U32 rIdx = threadIdx.x * 4u;         // 4 groups per lane
      U32 rRow = rIdx >> logBankSize;
      U32 rCol = rIdx & BankMask;

      U32 rowIdx = (rRow * strideBank) + (rCol + 1u);
         // Extra '+1' to shift past initial pad element      

      U32 * rowPtr = &s_thrdCounts[rowIdx];

      // Store row counts in row array
      rowPtr[0] = rowCnt1;
      rowPtr[1] = rowCnt2;
      rowPtr[2] = rowCnt3;
      rowPtr[3] = rowCnt4;

      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }

      // Get Warp Row & Column
      U32 warpRow = threadIdx.x >> logWarpSize; // tid / 32
      U32 warpCol = threadIdx.x & WarpMask;     // tid % 32

      // Get local & global indices
      U32 outGlobal = (blockIdx.x * nHistBins);
      U32 outLocal  = (warpRow * OutWarpSize);
      U32 rowBase   = (warpRow * OutStrideSize);
      U32 outBase   = outGlobal + outLocal;
      U32 rowOff    = warpCol + 1u;

      U32 outIdx = outBase + warpCol;
          rowIdx = rowBase + rowOff;

      // Get local & global pointers
      U32 * outPtr = &outRowCounts[outIdx];
            rowPtr = &s_thrdCounts[rowIdx];

         // Write our 'per row' counts in warp sequential order
      if (OutLength >= 1u) { outPtr[(0u*WarpSize)] = rowPtr[(0u*strideBank)]; }
      if (OutLength >= 2u) { outPtr[(1u*WarpSize)] = rowPtr[(1u*strideBank)]; }
      if (OutLength >= 3u) { outPtr[(2u*WarpSize)] = rowPtr[(2u*strideBank)]; }
      if (OutLength >= 4u) { outPtr[(3u*WarpSize)] = rowPtr[(3u*strideBank)]; }
      if (OutLength >= 5u) { outPtr[(4u*WarpSize)] = rowPtr[(4u*strideBank)]; }
      if (OutLength >= 6u) { outPtr[(5u*WarpSize)] = rowPtr[(5u*strideBank)]; }
      if (OutLength >= 7u) { outPtr[(6u*WarpSize)] = rowPtr[(6u*strideBank)]; }
      if (OutLength >= 8u) { outPtr[(7u*WarpSize)] = rowPtr[(7u*strideBank)]; }
   }
}



//-----------------------------------------------
// Name: K2_TRISH_RowCounts_To_RowStarts
// Desc: Sum 256-way 'per row' counts into 
//       total 256-way counts using prefix-sum
//------------------------------------------------

template < 
           U32 logBankSize,		// log<2>( Channels per Bank )
           U32 logWarpSize,		// log<2>( Threads Per Warp )
           U32 BlockSize 	      // Threads Per Block
         >
__global__
void K2_TRISH_RowCounts_To_RowStarts
( 
         U32 * outTotalCounts,	// OUT - total counts
         U32 * outTotalStarts,	// OUT - total starts
         U32 * outRowStarts,	   // OUT - row starts
	const U32 * inRowCounts,	   // IN  - 'per row' counts to accumulate
         U32   nRows			      // IN  - number of rows to accumulate
) 
{
	//------------------------------------
	// Constant values
	//------------------------------------

		// Memory Channels Per Bank
	const U32 BankSize  = 1u << logBankSize;	// 32 (or 16)
	const U32 BankMask  = BankSize - 1u;	   // 31 (or 15)

		// Threads Per Warp
	const U32 WarpSize  = 1u << logWarpSize;	// 32
	const U32 WarpMask  = WarpSize - 1u;      // 31

		// Warps Per Block
	const U32 WarpsPerBlock = BlockSize / WarpSize; // 8 = 256 / 32
	
		// Size of 'Row Counts' and 'Row Starts' array
	//const U32 nElemsCounts = 256;
	//const U32 banksCounts  = (nElemsCounts + BankMask) / BankSize;
	//const U32 padCounts    = ((banksCounts * BankSize) - nElemsCounts);
	//const U32 sizeCounts   = nElemsCounts + padCounts;

      // Stride for padded bank of elements
   const U32 strideBank = 1u + BankSize;

		// Serial Scan Array
   const U32 nSS1      = 256u + 2u;
   const U32 nRowsSS1  = (nSS1 + BankMask) / BankSize;
	const U32 nElemsSS1 = nRowsSS1 * strideBank;
	const U32 banksSS1  = (nElemsSS1 + BankMask) / BankSize;
	const U32 padSS1    = ((banksSS1 * BankSize) - nElemsSS1);
	const U32 sizeSS1   = nElemsSS1 + padSS1;

		// WarpScan array
	const U32 strideWS2 = WarpSize
		                   + (WarpSize >> 1u)
						       + 1u;			// 49 = (32 + 16 + 1)
   const U32 nWarpsWS2 = 1u;
	const U32 nElemsWS2 = nWarpsWS2 * strideWS2;
	const U32 banksWS2  = (nElemsWS2 + BankMask) / BankSize;
	const U32 padWS2    = ((banksWS2 * BankSize) - nElemsWS2);
	const U32 sizeWS2   = nElemsWS2 + padWS2;

	//const U32 nSafePassesCnts = sizeCounts / BlockSize;
	//const U32 leftOverCnts    = sizeCounts - (nSafePassesCnts * BlockSize);

	const U32 nSafePassesSS1  = sizeSS1 / BlockSize;
	const U32 leftOverSS1     = sizeSS1 - (nSafePassesSS1 * BlockSize);

	const U32 nSafePassesWS2  = sizeWS2 / BlockSize;
	const U32 leftOverWS2     = sizeWS2 - (nSafePassesWS2 * BlockSize);


	//------------------------------------
	// Local variables
	//------------------------------------

		// shared memory
	//__shared__ U32 s_rowStarts[sizeCounts];	// 'Row Starts' one chunk at a time
   __shared__ U32 s_ss1[sizeSS1];            // Used for serial scan
	__shared__ U32 s_ws2[sizeWS2];		      // Used for parallel warp scan

		// Registers
	U32 tSum;				// Per thread accumulator


	//------------------------------------
	// Compute Indices & Pointers
	//------------------------------------

   U32 warpRow, warpCol;
   U32 storeIdx, prevIdx, ss1Idx, ws2Idx;
   {
      // Compute Bank Offsets
	   //U32 bankRow = threadIdx.x >> logBankSize;		// tid / 32
	   U32 bankCol = threadIdx.x & BankMask;			// tid % 32

	   // Compute warp offsets
	   warpRow = threadIdx.x >> logWarpSize;		// tid / 32
	   warpCol = threadIdx.x & WarpMask;			// tid % 32

      // Compute Store index (for storing final counts before prefix sum)
      U32 sIdx = threadIdx.x;
      U32 storeRow = sIdx >> logBankSize;   // tid / 32
      U32 storeCol = sIdx & BankMask;       // tid % 32
      storeIdx = (storeRow * strideBank)
                 + storeCol
                 + 2u;        // Pad for 'reach back'

	      //--
	      // Previous Column (Serial Scan 1)
	      //   1.) Reach back one column
	      //   2.) But, we need to skip over extra padding before the first
         //       thread in every bank, so reach back two columns
         // However, the very first thread in the very first bank needs
         // to be able to reach back safely 2 columns without going 'out of range'.
         //
         // We work around this by pre-padding the 's_ss1' array with
         // an extra 2 elements and shifting indices over by two as needed to skip over padding.
	      //--

 	   U32 prevCol = ((bankCol == 0u) ? 2u : 1u);
      prevIdx = storeIdx - prevCol;

      // Compute Serial Scan index
      U32 ssIdx  = threadIdx.x * 8u;
      U32 ss1Row = ssIdx >> logBankSize;   // (tid*8) / 32
      U32 ss1Col = ssIdx & BankMask;       // (tid*8) % 32
      ss1Idx = (ss1Row * strideBank)
               + ss1Col
               + 2u;       // pad for 'reach back'

	   // Compute Warp Scan Index
	   ws2Idx  = (warpRow * strideWS2) 
		          + (WarpSize >> 1u)
		          + warpCol;
	}


	//------------------------------------
	// Zero out 'arrays'
	//------------------------------------

   U32 * setPtr = NULL;

	//-
	// Zero out 'row starts' array
	//-

   //setPtr = (&s_rowStarts[0]);
   //SetArray_BlockSeq
   //   < 
   //      U32, BlockSize, nSafePassesCnts, 
   //      leftOverCnts, sizeCounts 
   //   >
   //   ( 
   //      setPtr, 0u
   //   );


   //-
	// Zero out 'Serial Scan' array
	//-

   setPtr = (&s_ss1[0]);
   SetArray_BlockSeq
      < 
         U32, BlockSize, nSafePassesSS1, 
         leftOverSS1, sizeSS1 
      >
      ( 
         setPtr, 0u
      );


   //-
	// Zero out 'Warp Scan' array
	//-

   setPtr = (&s_ws2[0]);
   SetArray_BlockSeq
      < 
         U32, BlockSize, nSafePassesWS2, 
         leftOverWS2, sizeWS2 
      >
      ( 
         setPtr, 0u
      );


   // Sync all threads in block
   if (WarpsPerBlock > 2u) { __syncthreads(); }


	//-------------------------------------------------
   // Phase 1:
	//   Serial Reduction of all rows of 'per row' counts
	//	  down to single set of 'total' counts
	//-------------------------------------------------

   {
      const U32 * inPtr = &inRowCounts[threadIdx.x];

	   // Initialize 'Thread Sum' to identity value
	   tSum = 0;

	   // Loop over row counts
	   #pragma unroll
	   for (U32 currPass = 0u; currPass < nRows; currPass++)
	   {		
		   // Grab count from global arrary
		   U32 currCnt = inPtr[0];

		   // Accumulate 'per row' counts into a 'total' count
		   tSum = tSum + currCnt;

		   // Move to next set of 'row counts' to process
         inPtr += BlockSize;
	   }

	   // Store the 'total count's
	   outTotalCounts[threadIdx.x] = tSum;

	   // Also store 'total count's into 'Serial Scan' array
      s_ss1[storeIdx] = tSum;

      // Sync all threads in block
      if (WarpsPerBlock > 2u) { __syncthreads(); }
   }


	//--------------------------------------
   // Phase 2:
	//   convert 'total counts' into 'total starts'
   //   using prefix sum
   //--------------------------------------

   if (warpRow == 0)
   {
	   volatile U32 * wsPtr = (U32 *)&(s_ws2[0]);
   	
      U32 * SS1_ptr = &s_ss1[ss1Idx];

		   // For higher performance, we use registers instead of shared memory
		   // Tradeoff - lots of register pressure (8 registers per thread)
      U32 ss01, ss02, ss03, ss04;
      U32 ss05, ss06, ss07, ss08;

      //-----
      // Serial Scan (on short sequence of 8 values)
      //-----

      // Grab short sequence of 8 values from ss1 array
      ss01 = SS1_ptr[0];
      ss02 = SS1_ptr[1];
      ss03 = SS1_ptr[2];
      ss04 = SS1_ptr[3];
      ss05 = SS1_ptr[4];
      ss06 = SS1_ptr[5];
      ss07 = SS1_ptr[6];
      ss08 = SS1_ptr[7];

      // Serial scan short sequence (in registers)
      //ss01 = <identity> + ss01;
      ss02 = ss01 + ss02;
      ss03 = ss02 + ss03;
      ss04 = ss03 + ss04;
      ss05 = ss04 + ss05;
      ss06 = ss05 + ss06;
      ss07 = ss06 + ss07;
      ss08 = ss07 + ss08;

      //-
      // Store final serial scan result into warp scan array
      //-

      U32 wi = ws2Idx;
      tSum = ss08;
      wsPtr[wi] = tSum;

	   //-----
	   // Warp Scan (on 32 threads in parallel)
	   //-----

      wsPtr[wi] = tSum = wsPtr[wi -  1u] + tSum;
      wsPtr[wi] = tSum = wsPtr[wi -  2u] + tSum;
      wsPtr[wi] = tSum = wsPtr[wi -  4u] + tSum;
      wsPtr[wi] = tSum = wsPtr[wi -  8u] + tSum;
      wsPtr[wi] = tSum = wsPtr[wi - 16u] + tSum;


      //-----
      // Serial Update (on short sequence of 8 values)
      //-----

      //-
      // Grab update (prefix) value from Warp Array
      //-
         // Note:  Need to reach back 'one column' to get exclusive result
      U32 prevWI = wi - 1u;
      tSum = wsPtr[prevWI];


      //-
      // Update each element short sequence with prefix (in registers)
      //-

      ss01 = tSum + ss01;
      ss02 = tSum + ss02;
      ss03 = tSum + ss03;
      ss04 = tSum + ss04;
      ss05 = tSum + ss05;
      ss06 = tSum + ss06;
      ss07 = tSum + ss07;
      ss08 = tSum + ss08;

      // Store 'prefix sum' results back in 'serial scan' array
      SS1_ptr[0] = ss01;
      SS1_ptr[1] = ss02;
      SS1_ptr[2] = ss03;
      SS1_ptr[3] = ss04;
      SS1_ptr[4] = ss05;
      SS1_ptr[5] = ss06;
      SS1_ptr[6] = ss07;
      SS1_ptr[7] = ss08;
   } // end warpRow == 0

   // Sync all threads in block
   if (WarpsPerBlock > 2u) { __syncthreads(); }


   //-----
   // Grab starting 'row start' (total sum) for this thread
   //    Note #1:  Need to 'reach back' one column for exclusive results
   //    Note #2:  This will result in an unavoidable '2-way' bank conflict
   //-----

   U32 rowSum = s_ss1[prevIdx];

	// Store total starts (from previous column)
	outTotalStarts[threadIdx.x] = rowSum;

   // Sync all threads in block
   if (WarpsPerBlock > 2u) { __syncthreads(); }


	//-------------------------------------------------
   // Phase 3:
   //    Accumulate and write out 'per row' starts
	//-------------------------------------------------

   {
      const U32 * inPtr  = &inRowCounts[threadIdx.x];
            U32 * outPtr = &outRowStarts[threadIdx.x];

	   // Initialize 'Thread Sum' to identity value

	   // Loop over row counts
	   #pragma unroll
	   for (U32 currPass = 0u; currPass < nRows; currPass++)
	   {		
		   // Read 'in' current count from global arrary
		   U32 currCnt = inPtr[0];

         // Write 'out' current row sum to global array
         outPtr[0] = rowSum;

		   // Accumulate 'per row' count into running 'row sum' start
		   rowSum = rowSum + currCnt;

         //-
		   // Move to next row
         //-
         
         inPtr  += BlockSize;
         outPtr += BlockSize;
	   }
      // Sync all threads in block
      //if (WarpsPerBlock > 2u) { __syncthreads(); }
   }
}


////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU TRISH histogram
////////////////////////////////////////////////////////////////////////////////


/*-----------------
  Local Defines
-----------------*/

// Number of SM's per GPU
#if (GPU_GTX_560M == GPU_PLATFORM)
   #define NUM_GPU_SMs (4u)
#elif (GPU_TELSA_M2050 == GPU_PLATFORM)
   #define NUM_GPU_SMs (14u)
#elif (GPU_GTX_480 == GPU_PLATFORM)
   #define NUM_GPU_SMs (15u)
#elif (GPU_GTX_580 == GPU_PLATFORM)
   #define NUM_GPU_SMs (16u)
#elif (GPU_GTX_680 == GPU_PLATFORM)
   #define NUM_GPU_SMs (8u)
#else
   // Unknown GPU - assume 16 SM's for now...
   #define NUM_GPU_SMs (16u)
#endif


// Intermediate CUDA buffers
static U32 * d_rowCounts = NULL;
static U32 * d_rowStarts = NULL;
static U32 * d_totalStarts = NULL;


//-----------------------------------------------
// Name:  initTrish256
// Desc:  Initialize intermediate GPU Buffers
//-----------------------------------------------

extern "C" 
void initTrish256( void )
{
	// Local Constants
	const U32 nHistBins256  = 256u;
	const U32 nGPU_SMs      = NUM_GPU_SMs;
	const U32 nGPU_ConcurrentBlocks = 3u;
	const U32 K1_GridSize   = nGPU_SMs * nGPU_ConcurrentBlocks;
	const U32 K1_nRows      = K1_GridSize;
	const U32 sizeRowCounts = K1_nRows * nHistBins256 * sizeof(U32);
	const U32 sizeTotal     = nHistBins256 * sizeof(U32);

	// Create intermediate GPU buffers
   cutilSafeCall( cudaMalloc( (void **)&d_rowCounts, sizeRowCounts ) );
   cutilSafeCall( cudaMalloc( (void **)&d_rowStarts, sizeRowCounts ) );
   cutilSafeCall( cudaMalloc( (void **)&d_totalStarts, sizeTotal ) );
}



//-----------------------------------------------
// Name:  closeTrish256
// Desc:  cleanup intermediate GPU buffers
//-----------------------------------------------

extern "C" 
void closeTrish256( void )
{
	// Destroy Intermediate GPU buffers
   cutilSafeCall( cudaFree( d_totalStarts ) );
	cutilSafeCall( cudaFree( d_rowStarts ) );
	cutilSafeCall( cudaFree( d_rowCounts ) );
}



//---------------------------------------------------------
// Name:  genTrishByteU8
// Desc:  CPU Wrapper function around 
//        generalized TRISH histogram for byte data
//        invoked by "genTRISH" demo
//---------------------------------------------------------

extern "C" 
void genTrishByteU8
(
	// Function Parameters
   U32  * d_Histogram,	// OUT - Final 256-way histogram counts
   void * d_Data,		   //  IN - input data to bin & count into histogram
   U32    byteCount,		//  IN - length of input data array
   U32    minVal,       //  IN - minVal
   U32    maxVal,       //  IN - maxVal
   U32    numBins       //  IN - number of bins
)
{
	//-----
	// Local Constants=
	//-----

      // Note:  The best # of blocks for the TRISH algorithm appears to be
      //        The # of SM's on the card * the number of concurrent blocks.
      //        This is the mininum to effectively use all hardware resources effectively.
      // 
      // For Example:  On the following Fermi cards, the grid sizes for best performance would be ... 
      //  GTX 560M    = 12 =  4 * 3
      //  TELSA M2050 = 42 = 14 * 3
      //  GTX 480     = 45 = 15 * 3
      //  GTX 580     = 48 = 16 * 3

	const U32 nGPU_SMs     = NUM_GPU_SMs;	// See #defines above
	const U32 nGPU_ConcurrentBlocks = 3u;	// for Fermi architectures, we can achieve 3 concurrent blocks per SM (64 * 3 = 192 => 192/1536 => 12.5% occupancy 
	const U32 logBankSize  = 5u;		      //  5 = log<2>( Memory Banks )
	const U32 logWarpSize  = 5u;           //  5 = log<2>( Threads per Warp )
	
	const U32 K1_BlockSize = 64u;          // 64 = Threads per Block (Histogram Kernel)
	const U32 K1_GridSize  = nGPU_SMs * nGPU_ConcurrentBlocks;	 // GridSize (Histogram Kernel)

	const U32 K2_BlockSize = 256u;		   // 256 = Threads per Block (RowSum Kernel)
	const U32 K2_GridSize  = 1u;		      //  1 = GridSize (RowSum Kernel)
	
	const U32 K1_Length    = 31u;		      //  31 = Work Per thread (loop unrolling)
	const U32 in_start     = 0u;		      //   0 = starting range
	const U32 K1_nRows     = K1_GridSize;	//  ?? = Number of rows (blocks) that are cooperatively striding across input data set


	//-----
	// Get number of elements
	//-----

   assert( byteCount > 0u );
   assert( byteCount % sizeof(U32) == 0u );

	U32 nElems = byteCount >> 2u;  // byteCount/4
	U32 in_stop = nElems - 1u;	

	const U32 * d_inVals = (const U32 *)d_Data;


	/*--------------------------------------
	  Step 0. Create Intermediate buffers 
    --------------------------------------*/

	// Code moved to initTrish256() above


	/*------------------------------------------------------
	  Step 1. Bin & count elements into 'per row' 256-way histograms
	------------------------------------------------------*/

   typedef MapToBin
            < 
               U32,     // Value Type
               F32,     // Conversion Type
               U32,     // Bin Type
               1u,      // Formula = #1: B=(A-Mu)*Alpha; where Mu = Min - 0.5; and Alpha = n/(max-min+1);
               1u,      // Range check for values <= min
               1u       // Range check for values >= max
            > MapperU8;


	K1_TRISH_CountRows_GEN_B1
		< 
		  // Template Parameters
        U32,            // underlying value type
        MapperU8,       // underlying mapper type
		  logBankSize,		// log<2>( Memory Banks ) 
		  logWarpSize,		// log<2>( Threads per Warp )
		  K1_BlockSize,	// Threads per Block
		  K1_GridSize,    // Blocks per Grid
		  K1_Length			// Work Per Thread (Loop unrolling)
		>
		<<< 
			// CUDA CTA Parameters
			K1_GridSize,	// Blocks per Grid 
			K1_BlockSize	// Threads per Block
		>>>
		(
			// Function parameters
			d_rowCounts,	// IN - 'per row' histograms
			d_inVals,		// IN - 'input' data to count & bin
			in_start,		// IN - input range [start, stop] 
			in_stop,			//      ditto
         minVal,        // IN - [min,max] value for histogram binning 
         maxVal,        //      ditto
         numBins        //      number of bins in histogram
		);
   // Check if kernel execution generated an error    
   cutilCheckMsg( "K1_TRISH_CountRows_GEN_B1() Kernel execution failed!" );


   /*------------------------------------------------------
	   Step 2. Sum 'per row' histograms into 'final' 256-bin histogram
   ------------------------------------------------------*/

   K2_TRISH_RowCounts_To_RowStarts
      < 
			// Template Parameters
			logBankSize,	// log<2>( Memory Banks ) 
			logWarpSize,	// log<2>( Warp Size )
			K2_BlockSize	// Threads per Block
		>
      <<< 
			// CUDA CTA Parameters
			K2_GridSize,	// Blocks per Grid 
			K2_BlockSize	// Threads per Block
		>>>	
      (
			// Function parameters
			d_Histogram,    // OUT - Histogram Counts
			d_totalStarts,  // OUT - Histogram Starts
			d_rowStarts,    // OUT - 'Per Row' Histogram Starts
			d_rowCounts,    // IN  - 'Per Row' Histogram Counts
			K1_nRows		// IN  - number of rows
      );
	// Check if kernel execution generated an error    
	cutilCheckMsg( "K2_TRISH_RowCounts_To_RowStarts() Kernel execution failed!" );


#if 1 == TRISH_VERIFY_HISTOGRAM

   //-----
	// Step 3. Verify Histogram results are correct
   //-----

	TRISH_VerifyHistogram_B1< U32, MyMapper >
      ( 
         nElems, (U32 *)d_inVals, 
         numBins, (U32 *)d_Histogram, 
         minVal, maxVal
      );
#endif


	/*--------------------------------------
	  Step 4. Cleanup intermediate buffers
	--------------------------------------*/

	// Code moved to closeTrish256() above
}


//---------------------------------------------------------
// Name:  genTrishWordU16
// Desc:  CPU Wrapper function around 
//        generalized TRISH histogram for word data
//        invoked by "genTRISH" demo
//---------------------------------------------------------

extern "C" 
void genTrishWordU16
(
	// Function Parameters
   U32  * d_Histogram,	// OUT - Final 256-way histogram counts
   void * d_Data,		   //  IN - input data to bin & count into histogram
   U32    wordCount,		//  IN - length of input data array
   U32    minVal,       //  IN - minVal
   U32    maxVal,       //  IN - maxVal
   U32    numBins       //  IN - number of bins
)
{
	//-----
	// Local Constants=
	//-----

      // Note:  The best # of blocks for the TRISH algorithm appears to be
      //        The # of SM's on the card * the number of concurrent blocks.
      //        This is the mininum to effectively use all hardware resources effectively.
      // 
      // For Example:  On the following Fermi cards, the grid sizes for best performance would be ... 
      //  GTX 560M    = 12 =  4 * 3
      //  TELSA M2050 = 42 = 14 * 3
      //  GTX 480     = 45 = 15 * 3
      //  GTX 580     = 48 = 16 * 3

	const U32 nGPU_SMs     = NUM_GPU_SMs;	// See #defines above
	const U32 nGPU_ConcurrentBlocks = 3u;	// for Fermi architectures, we can achieve 3 concurrent blocks per SM (64 * 3 = 192 => 192/1536 => 12.5% occupancy 
	const U32 logBankSize  = 5u;		      //  5 = log<2>( Memory Banks )
	const U32 logWarpSize  = 5u;           //  5 = log<2>( Threads per Warp )
	
	const U32 K1_BlockSize = 64u;          // 64 = Threads per Block (Histogram Kernel)
	const U32 K1_GridSize  = nGPU_SMs * nGPU_ConcurrentBlocks;	 // GridSize (Histogram Kernel)

	const U32 K2_BlockSize = 256u;		   // 256 = Threads per Block (RowSum Kernel)
	const U32 K2_GridSize  = 1u;		      //  1 = GridSize (RowSum Kernel)

   // Efficiency Formula = (Floor(127/k)*k)/127
      // Ideal k-values = 1, 127 (IE efficiency = 1)
      // Best k-values otherwise = 2,3,6,7,9,14,18,21,42,63
      // Also try 25 & 31 (Local Maxima)
      // Worst k-values = 50 (0.504) and 43 (0.677) and 32 (0.756)
	//const U32 K1_Length    =   1u;		      //    1, Efficiency = 1.0,   Throughput = 17.75 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   2u;		      //    2, Efficiency = 0.992, Throughput = 27.20 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   3u;		      //    3, Efficiency = 0.992, Throughput = 32.71 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   6u;		      //    6, Efficiency = 0.992, Throughput = 40.02 GB/s (480)
	//const U32 K1_Length    =   7u;		      //    7, Efficiency = 0.992, Throughput = 41.50 GB/s (480)
	//const U32 K1_Length    =   9u;		      //    9, Efficiency = 0.992, Throughput = 40.21 GB/s (480)
	//const U32 K1_Length    =  14u;		      //   14, Efficiency = 0.992, Throughput = 43.56 GB/s (480)
	//const U32 K1_Length    =  18u;		      //   18, Efficiency = 0.992, Throughput = 44.08 GB/s (480)
	//const U32 K1_Length    =  21u;		      //   21, Efficiency = 0.992, Throughput = 43.74 GB/s (480)
	//const U32 K1_Length    =  25u;		      //   25, Efficiency = 0.984, Throughput = 44.21 GB/s (480)
	//const U32 K1_Length    =  31u;		      //   31, Efficiency = 0.976, Throughput = 44.29 GB/s (480)
	//const U32 K1_Length    =  42u;		      //   42, Efficiency = 0.992, Throughput = 44.43 GB/s (480)
	const U32 K1_Length    =  63u;		      //   63, Efficiency = 0.992, Throughput = 45.66 GB/s (480), *BEST* result
   //const U32 K1_Length    =  64u;          //   64, Efficiency = 0.504, Throughput = 41.10 GB/s (480),  *WORST* Efficiency
	//const U32 K1_Length    = 105u;          //  106, Efficiency = 0.827, Throughput = 44.86 GB/s (480), Good result, Program probably still fits in code cache...
	//const U32 K1_Length    = 106u;          //  106, Efficiency = 0.835, Throughput = 42.60 GB/s (480), Starts declining, ??? Program too large to fit in code cache ???
	//const U32 K1_Length    = 127u;          //  127, Efficiency = 1.0,   Throughput = 26.16 GB/s (480), *POOR* performance, ??? Program too large to fit in code cache ???
	const U32 in_start     = 0u;		      //   0 = starting range
	const U32 K1_nRows     = K1_GridSize;	//  ?? = Number of rows (blocks) that are cooperatively striding across input data set


	//-----
	// Get number of elements
	//-----

   assert( wordCount > 0u );
   assert( wordCount % 4 == 0u );

	U32 nElems = wordCount >> 1u;  // wordCount/2
	U32 in_stop = nElems - 1u;	

	const U32 * d_inVals = (const U32 *)d_Data;


	/*--------------------------------------
	  Step 0. Create Intermediate buffers 
    --------------------------------------*/

	// Code moved to initTrish256() above


	/*------------------------------------------------------
	  Step 1. Bin & count elements into 'per row' 256-way histograms
	------------------------------------------------------*/

   typedef MapToBin
            < 
               U32,     // Value Type
               F32,     // Conversion Type
               U32,     // Bin Type
               1u,      // Formula = #1: B=(A-Mu)*Alpha; where Mu = Min - 0.5; and Alpha = n/(max-min+1);
               1u,      // Range check for values < min
               1u       // Range check for values > max
            > MapperU16;


	K1_TRISH_CountRows_GEN_B2
		< 
		  // Template Parameters
        U32,            // underlying value type
        MapperU16,      // underlying mapper type
		  logBankSize,		// log<2>( Memory Banks ) 
		  logWarpSize,		// log<2>( Threads per Warp )
		  K1_BlockSize,	// Threads per Block
		  K1_GridSize,    // Blocks per Grid
		  K1_Length			// Work Per Thread (Loop unrolling)
		>
		<<< 
			// CUDA CTA Parameters
			K1_GridSize,	// Blocks per Grid 
			K1_BlockSize	// Threads per Block
		>>>
		(
			// Function parameters
			d_rowCounts,	// IN - 'per row' histograms
			d_inVals,		// IN - 'input' data to count & bin
			in_start,		// IN - input range [start, stop] 
			in_stop,			//      ditto
         minVal,        // IN - [min,max] value for histogram binning 
         maxVal,        //      ditto
         numBins        //      number of bins in histogram
		);
   // Check if kernel execution generated an error    
   cutilCheckMsg( "K1_TRISH_CountRows_GEN_B2() Kernel execution failed!" );


   /*------------------------------------------------------
	   Step 2. Sum 'per row' histograms into 'final' 256-bin histogram
   ------------------------------------------------------*/

   K2_TRISH_RowCounts_To_RowStarts
      < 
			// Template Parameters
			logBankSize,	// log<2>( Memory Banks ) 
			logWarpSize,	// log<2>( Warp Size )
			K2_BlockSize	// Threads per Block
		>
      <<< 
			// CUDA CTA Parameters
			K2_GridSize,	// Blocks per Grid 
			K2_BlockSize	// Threads per Block
		>>>	
      (
			// Function parameters
			d_Histogram,    // OUT - Histogram Counts
			d_totalStarts,  // OUT - Histogram Starts
			d_rowStarts,    // OUT - 'Per Row' Histogram Starts
			d_rowCounts,    // IN  - 'Per Row' Histogram Counts
			K1_nRows		// IN  - number of rows
      );
	// Check if kernel execution generated an error    
	cutilCheckMsg( "K2_TRISH_RowCounts_To_RowStarts() Kernel execution failed!" );


#if 1 == TRISH_VERIFY_HISTOGRAM

   //-----
	// Step 3. Verify Histogram results are correct
   //-----

	TRISH_VerifyHistogram_B2< U32, MyMapper >
      ( 
         nElems, (U32 *)d_inVals, 
         numBins, (U32 *)d_Histogram, 
         minVal, maxVal
      );
#endif


	/*--------------------------------------
	  Step 4. Cleanup intermediate buffers
	--------------------------------------*/

	// Code moved to closeTrish256() above
}


//---------------------------------------------------------
// Name:  genTrishDWordU32
// Desc:  CPU Wrapper function around 
//        generalized TRISH histogram for DWORD data
//        invoked by "genTRISH" demo
//---------------------------------------------------------

extern "C" 
void genTrishDWordU32
(
	// Function Parameters
   U32  * d_Histogram,	// OUT - Final 256-way histogram counts
   void * d_Data,		   //  IN - input data to bin & count into histogram
   U32    dwordCount,	//  IN - length of input data array
   U32    minVal,       //  IN - minVal
   U32    maxVal,       //  IN - maxVal
   U32    numBins       //  IN - number of bins
)
{
	//-----
	// Local Constants=
	//-----

      // Note:  The best # of blocks for the TRISH algorithm appears to be
      //        The # of SM's on the card * the number of concurrent blocks.
      //        This is the mininum to effectively use all hardware resources effectively.
      // 
      // For Example:  On the following Fermi cards, the grid sizes for best performance would be ... 
      //  GTX 560M    = 12 =  4 * 3
      //  TELSA M2050 = 42 = 14 * 3
      //  GTX 480     = 45 = 15 * 3
      //  GTX 580     = 48 = 16 * 3

	const U32 nGPU_SMs     = NUM_GPU_SMs;	// See #defines above
	const U32 nGPU_ConcurrentBlocks = 3u;	// for Fermi architectures, we can achieve 3 concurrent blocks per SM (64 * 3 = 192 => 192/1536 => 12.5% occupancy 
	const U32 logBankSize  = 5u;		      //  5 = log<2>( Memory Banks )
	const U32 logWarpSize  = 5u;           //  5 = log<2>( Threads per Warp )
	
	const U32 K1_BlockSize = 64u;          // 64 = Threads per Block (Histogram Kernel)
	const U32 K1_GridSize  = nGPU_SMs * nGPU_ConcurrentBlocks;	 // GridSize (Histogram Kernel)

	const U32 K2_BlockSize = 256u;		   // 256 = Threads per Block (RowSum Kernel)
	const U32 K2_GridSize  = 1u;		      //  1 = GridSize (RowSum Kernel)

   // Efficiency Formula = (Floor(255/k)*k)/255
      // Ideal k-values = 1, 3, 5, 15, 17, 51, 85, 255 
      // Best k-values otherwise = 2, 4, 6, 7, 9, 10, 11, 12, 14, 18, 21, 23, 25, 28, 36, 42, 50, 63, 84, 125, 126, 127, 253, 254
      // Worst k-values = 128 (0.502) & 86 (0.675) & 64 (0.753) 
      // K >= 105 => code won't fit in cache
	//const U32 K1_Length    =   1u;		      //    1, Efficiency = 1.0,   Throughput = 19.66 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   2u;		      //    2, Efficiency = 0.996, Throughput = 34.16 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   3u;		      //    3, Efficiency = 1.0,   Throughput = 44.90 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   4u;		      //    4, Efficiency = 0.988, Throughput = 52.03 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   5u;		      //    5, Efficiency = 1.0,   Throughput = 56.56 GB/s (480),
	//const U32 K1_Length    =   6u;		      //    6, Efficiency = 0.988, Throughput = 60.32 GB/s (480)
	//const U32 K1_Length    =   7u;		      //    7, Efficiency = 0.988, Throughput = 53.07 GB/s (480)
	//const U32 K1_Length    =   9u;		      //    9, Efficiency = 0.988, Throughput = 59.97 GB/s (480)
	//const U32 K1_Length    =  10u;		      //   10, Efficiency = 0.980, Throughput = 61.61 GB/s (480)
	//const U32 K1_Length    =  11u;		      //   11, Efficiency = 0.992, Throughput = 62.57 GB/s (480)
	//const U32 K1_Length    =  12u;		      //   12, Efficiency = 0.988, Throughput = 62.00 GB/s (480)
	//const U32 K1_Length    =  14u;		      //   14, Efficiency = 0.988, Throughput = 64.24 GB/s (480)
	const U32 K1_Length    =  15u;		      //   15, Efficiency = 1.0,   Throughput = 65.05 GB/s (480)  *BEST*
	//const U32 K1_Length    =  16u;		      //   16, Efficiency = 0.941, Throughput = 63.14 GB/s (480)
	//const U32 K1_Length    =  17u;		      //   17, Efficiency = 1.0,   Throughput = 63.06 GB/s (480)
	//const U32 K1_Length    =  18u;		      //   18, Efficiency = 0.988, Throughput = 58.58 GB/s (480)
	//const U32 K1_Length    =  21u;		      //   21, Efficiency = 0.988, Throughput = 59.07 GB/s (480)
	//const U32 K1_Length    =  23u;		      //   23, Efficiency = 0.992, Throughput = 59.99 GB/s (480)
	//const U32 K1_Length    =  25u;		      //   25, Efficiency = 0.980, Throughput = 61.24 GB/s (480)
	//const U32 K1_Length    =  28u;		      //   28, Efficiency = 0.988, Throughput = 62.17 GB/s (480)
	//const U32 K1_Length    =  36u;		      //   36, Efficiency = 0.988, Throughput = 58.93 GB/s (480)
	//const U32 K1_Length    =  42u;		      //   42, Efficiency = 0.988, Throughput = 60.09 GB/s (480)
	//const U32 K1_Length    =  50u;		      //   50, Efficiency = 0.980, Throughput = 62.01 GB/s (480)
	//const U32 K1_Length    =  51u;		      //   51, Efficiency = 1.0,   Throughput = 62.46 GB/s (480)
	//const U32 K1_Length    =  63u;		      //   63, Efficiency = 0.988, Throughput = 62.88 GB/s (480),
   //const U32 K1_Length    =  84u;          //   84, Efficiency = 0.988, Throughput = 64.62 GB/s (480),
   //const U32 K1_Length    =  85u;          //   85, Efficiency = 1.0,   Throughput = 64.17 GB/s (480),
   //const U32 K1_Length    =  86u;          //   86, Efficiency = 0.675, Throughput = 60.61 GB/s (480), *POOR EFFICIENCY*
   //const U32 K1_Length    = 125u;          //  125, Efficiency = 0.980, Throughput = 65.41 GB/s (480), *BEST*
   //const U32 K1_Length    = 126u;          //  126, Efficiency = 0.988, Throughput = 65.55 GB/s (480), *BEST of the BEST*
   //const U32 K1_Length    = 127u;          //  127, Efficiency = 0.996, Throughput = 65.13 GB/s (480), *BEST*
   //const U32 K1_Length    = 128u;          //  128, Efficiency = 0.502, Throughput = 59.59 GB/s (480), *WORST EFFICIENCY*
   // K=[105..255], code probably won't fit in cache
	const U32 in_start     = 0u;		      //   0 = starting range
	const U32 K1_nRows     = K1_GridSize;	//  ?? = Number of rows (blocks) that are cooperatively striding across input data set


	//-----
	// Get number of elements
	//-----

   assert( dwordCount > 0u );
   assert( dwordCount % 4 == 0u );

	U32 nElems = dwordCount;
	U32 in_stop = nElems - 1u;	

	const U32 * d_inVals = (const U32 *)d_Data;


	/*--------------------------------------
	  Step 0. Create Intermediate buffers 
    --------------------------------------*/

	// Code moved to initTrish256() above


	/*------------------------------------------------------
	  Step 1. Bin & count elements into 'per row' 256-way histograms
	------------------------------------------------------*/

   typedef MapToBin
            < 
               U32,     // Value Type
               F32,     // Conversion Type
               U32,     // Bin Type
               1u,      // Formula = #1: B=(A-Mu)*Alpha; where Mu = Min - 0.5; and Alpha = n/(max-min+1);
               1u,      // Range check for values < min
               1u       // Range check for values > max
            > MapperU32;

	K1_TRISH_CountRows_GEN_B4
		< 
		  // Template Parameters
        U32,            // underlying value type
        MapperU32,      // underlying mapper type
		  logBankSize,		// log<2>( Memory Banks ) 
		  logWarpSize,		// log<2>( Threads per Warp )
		  K1_BlockSize,	// Threads per Block
		  K1_GridSize,    // Blocks per Grid
		  K1_Length			// Work Per Thread (Loop unrolling)
		>
		<<< 
			// CUDA CTA Parameters
			K1_GridSize,	// Blocks per Grid 
			K1_BlockSize	// Threads per Block
		>>>
		(
			// Function parameters
			d_rowCounts,	// IN - 'per row' histograms
			d_inVals,		// IN - 'input' data to count & bin
			in_start,		// IN - input range [start, stop] 
			in_stop,			//      ditto
         minVal,        // IN - [min,max] value for histogram binning 
         maxVal,        //      ditto
         numBins        //      number of bins in histogram
		);
   // Check if kernel execution generated an error    
   cutilCheckMsg( "K1_TRISH_CountRows_GEN_B2() Kernel execution failed!" );


   /*------------------------------------------------------
	   Step 2. Sum 'per row' histograms into 'final' 256-bin histogram
   ------------------------------------------------------*/

   K2_TRISH_RowCounts_To_RowStarts
      < 
			// Template Parameters
			logBankSize,	// log<2>( Memory Banks ) 
			logWarpSize,	// log<2>( Warp Size )
			K2_BlockSize	// Threads per Block
		>
      <<< 
			// CUDA CTA Parameters
			K2_GridSize,	// Blocks per Grid 
			K2_BlockSize	// Threads per Block
		>>>	
      (
			// Function parameters
			d_Histogram,    // OUT - Histogram Counts
			d_totalStarts,  // OUT - Histogram Starts
			d_rowStarts,    // OUT - 'Per Row' Histogram Starts
			d_rowCounts,    // IN  - 'Per Row' Histogram Counts
			K1_nRows		// IN  - number of rows
      );
	// Check if kernel execution generated an error    
	cutilCheckMsg( "K2_TRISH_RowCounts_To_RowStarts() Kernel execution failed!" );


#if 1 == TRISH_VERIFY_HISTOGRAM

   //-----
	// Step 3. Verify Histogram results are correct
   //-----

	TRISH_VerifyHistogram_B4< U32, MyMapper >
      ( 
         nElems, (U32 *)d_inVals, 
         numBins, (U32 *)d_Histogram, 
         minVal, maxVal
      );
#endif


	/*--------------------------------------
	  Step 4. Cleanup intermediate buffers
	--------------------------------------*/

	// Code moved to closeTrish256() above
}



//---------------------------------------------------------
// Name:  genTrishFloatF32
// Desc:  CPU Wrapper function around 
//        generalized TRISH histogram for FLOAT data
//        invoked by "genTRISH" demo
//---------------------------------------------------------

extern "C" 
void genTrishFloatF32
(
	// Function Parameters
   U32  * d_Histogram,	// OUT - Final 256-way histogram counts
   void * d_Data,		   //  IN - input data to bin & count into histogram
   U32    floatCount,	//  IN - length of input data array
   F32    minVal,       //  IN - minVal
   F32    maxVal,       //  IN - maxVal
   U32    numBins       //  IN - number of bins
)
{
	//-----
	// Local Constants=
	//-----

      // Note:  The best # of blocks for the TRISH algorithm appears to be
      //        The # of SM's on the card * the number of concurrent blocks.
      //        This is the mininum to effectively use all hardware resources effectively.
      // 
      // For Example:  On the following Fermi cards, the grid sizes for best performance would be ... 
      //  GTX 560M    = 12 =  4 * 3
      //  TELSA M2050 = 42 = 14 * 3
      //  GTX 480     = 45 = 15 * 3
      //  GTX 580     = 48 = 16 * 3

	const U32 nGPU_SMs     = NUM_GPU_SMs;	// See #defines above
	const U32 nGPU_ConcurrentBlocks = 3u;	// for Fermi architectures, we can achieve 3 concurrent blocks per SM (64 * 3 = 192 => 192/1536 => 12.5% occupancy 
	const U32 logBankSize  = 5u;		      //  5 = log<2>( Memory Banks )
	const U32 logWarpSize  = 5u;           //  5 = log<2>( Threads per Warp )
	
	const U32 K1_BlockSize = 64u;          // 64 = Threads per Block (Histogram Kernel)
	const U32 K1_GridSize  = nGPU_SMs * nGPU_ConcurrentBlocks;	 // GridSize (Histogram Kernel)

	const U32 K2_BlockSize = 256u;		   // 256 = Threads per Block (RowSum Kernel)
	const U32 K2_GridSize  = 1u;		      //  1 = GridSize (RowSum Kernel)

   // Efficiency Formula = (Floor(255/k)*k)/255
      // Ideal k-values = 1, 3, 5, 15, 17, 51, 85, 255 
      // Best k-values otherwise = 2, 4, 6, 7, 9, 10, 11, 12, 14, 18, 21, 23, 25, 28, 36, 42, 50, 63, 84, 125, 126, 127, 253, 254
      // Worst k-values = 128 (0.502) & 86 (0.675) & 64 (0.753) 
      // K >= 105 => code won't fit in cache
	//const U32 K1_Length    =   1u;		      //    1, Efficiency = 1.0,   Throughput = 19.66 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   2u;		      //    2, Efficiency = 0.996, Throughput = 34.16 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   3u;		      //    3, Efficiency = 1.0,   Throughput = 44.90 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   4u;		      //    4, Efficiency = 0.988, Throughput = 52.03 GB/s (480), *POOR ILP*
	//const U32 K1_Length    =   5u;		      //    5, Efficiency = 1.0,   Throughput = 56.56 GB/s (480),
	//const U32 K1_Length    =   6u;		      //    6, Efficiency = 0.988, Throughput = 60.32 GB/s (480)
	//const U32 K1_Length    =   7u;		      //    7, Efficiency = 0.988, Throughput = 53.07 GB/s (480)
	//const U32 K1_Length    =   9u;		      //    9, Efficiency = 0.988, Throughput = 59.97 GB/s (480)
	//const U32 K1_Length    =  10u;		      //   10, Efficiency = 0.980, Throughput = 61.61 GB/s (480)
	//const U32 K1_Length    =  11u;		      //   11, Efficiency = 0.992, Throughput = 62.57 GB/s (480)
	//const U32 K1_Length    =  12u;		      //   12, Efficiency = 0.988, Throughput = 62.00 GB/s (480)
	//const U32 K1_Length    =  14u;		      //   14, Efficiency = 0.988, Throughput = 64.24 GB/s (480)
	const U32 K1_Length    =  15u;		      //   15, Efficiency = 1.0,   Throughput = 65.05 GB/s (480)  *BEST*
	//const U32 K1_Length    =  16u;		      //   16, Efficiency = 0.941, Throughput = 63.14 GB/s (480)
	//const U32 K1_Length    =  17u;		      //   17, Efficiency = 1.0,   Throughput = 63.06 GB/s (480)
	//const U32 K1_Length    =  18u;		      //   18, Efficiency = 0.988, Throughput = 58.58 GB/s (480)
	//const U32 K1_Length    =  21u;		      //   21, Efficiency = 0.988, Throughput = 59.07 GB/s (480)
	//const U32 K1_Length    =  23u;		      //   23, Efficiency = 0.992, Throughput = 59.99 GB/s (480)
	//const U32 K1_Length    =  25u;		      //   25, Efficiency = 0.980, Throughput = 61.24 GB/s (480)
	//const U32 K1_Length    =  28u;		      //   28, Efficiency = 0.988, Throughput = 62.17 GB/s (480)
	//const U32 K1_Length    =  36u;		      //   36, Efficiency = 0.988, Throughput = 58.93 GB/s (480)
	//const U32 K1_Length    =  42u;		      //   42, Efficiency = 0.988, Throughput = 60.09 GB/s (480)
	//const U32 K1_Length    =  50u;		      //   50, Efficiency = 0.980, Throughput = 62.01 GB/s (480)
	//const U32 K1_Length    =  51u;		      //   51, Efficiency = 1.0,   Throughput = 62.46 GB/s (480)
	//const U32 K1_Length    =  63u;		      //   63, Efficiency = 0.988, Throughput = 62.88 GB/s (480),
   //const U32 K1_Length    =  84u;          //   84, Efficiency = 0.988, Throughput = 64.62 GB/s (480),
   //const U32 K1_Length    =  85u;          //   85, Efficiency = 1.0,   Throughput = 64.17 GB/s (480),
   //const U32 K1_Length    =  86u;          //   86, Efficiency = 0.675, Throughput = 60.61 GB/s (480), *POOR EFFICIENCY*
   //const U32 K1_Length    = 125u;          //  125, Efficiency = 0.980, Throughput = 65.41 GB/s (480), *BEST*
   //const U32 K1_Length    = 126u;          //  126, Efficiency = 0.988, Throughput = 65.55 GB/s (480), *BEST of the BEST*
   //const U32 K1_Length    = 127u;          //  127, Efficiency = 0.996, Throughput = 65.13 GB/s (480), *BEST*
   //const U32 K1_Length    = 128u;          //  128, Efficiency = 0.502, Throughput = 59.59 GB/s (480), *WORST EFFICIENCY*
   // K=[105..255], code probably won't fit in cache
	const U32 in_start     = 0u;		      //   0 = starting range
	const U32 K1_nRows     = K1_GridSize;	//  ?? = Number of rows (blocks) that are cooperatively striding across input data set


	//-----
	// Get number of elements
	//-----

   assert( floatCount > 0u );
   assert( floatCount % 4 == 0u );

	U32 nElems = floatCount;
	U32 in_stop = nElems - 1u;	

	const F32 * d_inVals = (const F32 *)d_Data;


	/*--------------------------------------
	  Step 0. Create Intermediate buffers 
    --------------------------------------*/

	// Code moved to initTrish256() above


	/*------------------------------------------------------
	  Step 1. Bin & count elements into 'per row' 256-way histograms
	------------------------------------------------------*/

   typedef MapToBin
            < 
               F32,     // Value Type
               F32,     // Conversion Type
               U32,     // Bin Type
               1u,      // Formula = #1: B=(A-Mu)*Alpha; where Mu = Min - 0.5; and Alpha = n/(max-min+1);
               0u,      // Range check for values < min
               0u       // Range check for values > max
            > MapperF32;

	K1_TRISH_CountRows_GEN_B4
		< 
		  // Template Parameters
        F32,            // underlying value type
        MapperF32,      // underlying mapper type
		  logBankSize,		// log<2>( Memory Banks ) 
		  logWarpSize,		// log<2>( Threads per Warp )
		  K1_BlockSize,	// Threads per Block
		  K1_GridSize,    // Blocks per Grid
		  K1_Length			// Work Per Thread (Loop unrolling)
		>
		<<< 
			// CUDA CTA Parameters
			K1_GridSize,	// Blocks per Grid 
			K1_BlockSize	// Threads per Block
		>>>
		(
			// Function parameters
			d_rowCounts,	// IN - 'per row' histograms
			d_inVals,		// IN - 'input' data to count & bin
			in_start,		// IN - input range [start, stop] 
			in_stop,			//      ditto
         minVal,        // IN - [min,max] value for histogram binning 
         maxVal,        //      ditto
         numBins        //      number of bins in histogram
		);
   // Check if kernel execution generated an error    
   cutilCheckMsg( "K1_TRISH_CountRows_GEN_B4() Kernel execution failed!" );


   /*------------------------------------------------------
	   Step 2. Sum 'per row' histograms into 'final' 256-bin histogram
   ------------------------------------------------------*/

   K2_TRISH_RowCounts_To_RowStarts
      < 
			// Template Parameters
			logBankSize,	// log<2>( Memory Banks ) 
			logWarpSize,	// log<2>( Warp Size )
			K2_BlockSize	// Threads per Block
		>
      <<< 
			// CUDA CTA Parameters
			K2_GridSize,	// Blocks per Grid 
			K2_BlockSize	// Threads per Block
		>>>	
      (
			// Function parameters
			d_Histogram,    // OUT - Histogram Counts
			d_totalStarts,  // OUT - Histogram Starts
			d_rowStarts,    // OUT - 'Per Row' Histogram Starts
			d_rowCounts,    // IN  - 'Per Row' Histogram Counts
			K1_nRows		// IN  - number of rows
      );
	// Check if kernel execution generated an error    
	cutilCheckMsg( "K2_TRISH_RowCounts_To_RowStarts() Kernel execution failed!" );


#if 1 == TRISH_VERIFY_HISTOGRAM

   //-----
	// Step 3. Verify Histogram results are correct
   //-----

	TRISH_VerifyHistogram_B4< F32, MyMapper >
      ( 
         nElems, (F32 *)d_inVals, 
         numBins, (U32 *)d_Histogram, 
         minVal, maxVal
      );
#endif


	/*--------------------------------------
	  Step 4. Cleanup intermediate buffers
	--------------------------------------*/

	// Code moved to closeTrish256() above
}
