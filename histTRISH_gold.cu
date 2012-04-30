/*-----------------------------------------------------------------------------
   Name: histTRISH_Gold.cu
   Desc: CPU Verification function to compare GPU results against
   
   Disclaimer:
      This software is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
  Includes
-----------------------------------------------------------------------------*/

// Standard Includes
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Local Includes
#include "Platform.h"
#include "BaseDefs.h"
#include "MapToBin.h"
#include "histogram_common.h"


/*-----------------------------------------------------------------------------
  Function Definitions
-----------------------------------------------------------------------------*/

/*-----------------------------
  Name: CPU_histogramGen_B1
  Desc: Verify CPU version of generic histogram code for byte data
-----------------------------*/

extern "C" void CPU_histogramGenByte_B1
(
    uint *h_Histogram,
    void *h_Data,
    uint byteCount,
    uint minVal,
    uint maxVal,
    uint nBins
)
{
    U32 testMod4 = (U32)byteCount % 4u;
    U32 count4   = (U32)byteCount / 4u;
    assert( (sizeof(U32) == 4u) && (testMod4 == 0u) );


    //-----
    // Zero out histogram array
    //-----
    
    for (U32 i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
    {
        h_Histogram[i] = 0u;
    }


    //-----
    // Bin all data elements
    //-----

    // Initialize Mapping Object
    typedef MapToBin
            < 
               U32,     // Value type
               F32,     // Conversion type
               U32,     // Bin type
               1u,      // Formula #1: B=(A-Mu)*Alpha; where Mu = (min-0.5); and Alpha = n/(max-min+1)
               1u,      // Range check values < min
               1u       // Range check values > max
            > MapperU8;
    MapperU8 mapper;
    mapper.Initiate( (U32)minVal, (U32)maxVal, (U32)nBins );

    U32 b1, b2, b3, b4;
    U32 bin1, bin2, bin3, bin4;
    U32 data;

    // Bin each element

    for (U32 i = 0u; i < count4; i++)
    {
        data = ((U32 *)h_Data)[i];

        // Extract 4 bytes from data
        b1 = data >>  0u;
        b2 = data >>  8u;
        b3 = data >> 16u;
        b4 = data >> 24u;

        b1 = b1 & 0xFFu;
        b2 = b2 & 0xFFu;
        b3 = b3 & 0xFFu;
        b4 = b4 & 0xFFu;

        // Transform from a single 32-bit storage item 
        // containing 4 8-bit bytes
        // to 4 bin indices under the mapping.        
        mapper.Transform4( bin1, bin2, bin3, bin4,
                           b1, b2, b3, b4 );

        // Bin results
        h_Histogram[bin1]++;
        h_Histogram[bin2]++;
        h_Histogram[bin3]++;
        h_Histogram[bin4]++;
    }

    
    // Cleanup mapping object
    mapper.Finish();

#if 0
    //-----
    // Print out CPU histogram counts
    //-----

    U32 total = 0u;
    for (U32 i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
    {
       U32 count = h_Histogram[i];
       total += count;
       fprintf( stdout, "Count[%u] = %u\n", i, count );
    }
    fprintf( stdout, "\nTotal = %u\n", total );
#endif
}


/*-----------------------------
  Name: CPU_histogramGenWord_B2
  Desc: Verify CPU version of generic histogram code for WORD data
-----------------------------*/

extern "C" void CPU_histogramGenWord_B2
(
    uint *h_Histogram,
    void *h_Data,
    uint byteCount,
    uint minVal,
    uint maxVal,
    uint nBins
)
{
    U32 dwordCount = (U32)byteCount / 4u;  // 4 bytes per dword (32-bit value)
    U32 testMod4   = (U32)byteCount % 4u;  // Make sure we DWORD aligned

    U32 nRows     = (dwordCount/2u) * 2u;  // Process two 32-bit values at a time
    U32 nLeftOver = dwordCount % 2u;       // Handle leftover values [0..1]
    assert( (sizeof(U32) == 4u) && (testMod4 == 0u) );


    //-----
    // Zero out histogram array
    //-----
    
    for (U32 i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
    {
        h_Histogram[i] = 0u;
    }


    //-----
    // Bin all data elements
    //-----

    // Initialize Mapping Object
    typedef MapToBin
            < 
               U32,     // Value type
               F32,     // Conversion type
               U32,     // Bin type
               1u,      // Formula #1: B=(A-Mu)*Alpha; where Mu = (min-0.5); and Alpha = n/(max-min+1)
               1u,      // Range check values < min
               1u       // Range check values > max
            > MapperU16;
    MapperU16 mapper;
    mapper.Initiate( (U32)minVal, (U32)maxVal, (U32)nBins );

    U32 b1, b2, b3, b4;
    U32 bin1, bin2, bin3, bin4;
    U32 data1, data2;

    // Bin each element

    U32 idx = 0u;
    for (idx = 0u; idx < nRows; idx += 2u)
    {
        // Grab 2 32-bit values
        data1 = ((U32 *)h_Data)[idx];
        data2 = ((U32 *)h_Data)[idx+1u];

        // Extract 4 words from two data values
        b1 = data1 & 0xFFFFu;
        b2 = data1 >> 16u;
        b3 = data2 & 0xFFFFu;
        b4 = data2 >> 16u;

        // Transform from 4 values to 4 bins
        mapper.Transform4( bin1, bin2, bin3, bin4,
                           b1, b2, b3, b4 );

        // Bin results
        h_Histogram[bin1]++;
        h_Histogram[bin2]++;
        h_Histogram[bin3]++;
        h_Histogram[bin4]++;
    }

    // Handle left over elements, if there any
    switch (nLeftOver)
    {
    case 0:
    default:
       // Do nothing
       break;

    case 1:
        // Process one left-over value
        data1 = ((U32 *)h_Data)[idx];

        // Extract 2 words from single value
        b1 = data1 & 0xFFFFu;
        b2 = data1 >> 16u;

        // Transform values to bins
        mapper.Transform2( bin1, bin2, b1, b2 );

        // Bin results
        h_Histogram[bin1]++;
        h_Histogram[bin2]++;
    } // end switch

    
    // Cleanup mapping object
    mapper.Finish();

#if 0
    //-----
    // Print out CPU histogram counts
    //-----

    U32 total = 0u;
    for (U32 i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
    {
       U32 count = h_Histogram[i];
       total += count;
       fprintf( stdout, "Count[%u] = %u\n", i, count );
    }
    fprintf( stdout, "\nTotal = %u\n", total );
#endif
}


/*-----------------------------
  Name: CPU_histogramGenDWord_B4
  Desc: Verify CPU version of generic histogram code for DWORD data
-----------------------------*/

extern "C" void CPU_histogramGenDWord_B4
(
    uint *h_Histogram,
    void *h_Data,
    uint byteCount,
    uint minVal,
    uint maxVal,
    uint nBins
)
{
    U32 dwordCount = (U32)byteCount / 4u; // 4 bytes per DWORD
    U32 testMod4 = (U32)byteCount % 4u;   // Make sure we are DWORD aligned
    assert( (sizeof(U32) == 4u) && (testMod4 == 0u) );

    U32 nRows     = (dwordCount/4u)*4u;  // Process 4 values at a time
    U32 nLeftOver = dwordCount % 4u;     // Process any leftover DWORDS [0..3]

    //-----
    // Zero out histogram array
    //-----
    
    for (U32 i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
    {
        h_Histogram[i] = 0u;
    }


    //-----
    // Bin all data elements
    //-----

    // Initialize Mapping Object
    typedef MapToBin
            < 
               U32,     // Value type
               F32,     // Conversion type
               U32,     // Bin type
               1u,      // Formula #1: B=(A-Mu)*Alpha; where Mu = (min-0.5); and Alpha = n/(max-min+1)
               1u,      // Range check values < min
               1u       // Range check values > max
            > MapperU32;
    MapperU32 mapper;
    mapper.Initiate( (U32)minVal, (U32)maxVal, (U32)nBins );

    U32 bin1, bin2, bin3, bin4;
    U32 data1, data2, data3, data4;

    // Bin each element
    U32 idx;
    for (idx = 0u; idx < nRows; idx += 4u)
    {
        data1 = ((U32 *)h_Data)[idx+0u];
        data2 = ((U32 *)h_Data)[idx+1u];
        data3 = ((U32 *)h_Data)[idx+2u];
        data4 = ((U32 *)h_Data)[idx+3u];

        // Transform from 4 values to 4 bins
        mapper.Transform4( bin1, bin2, bin3, bin4,
                           data1, data2, data3, data4 );

        // Bin results
        h_Histogram[bin1]++;
        h_Histogram[bin2]++;
        h_Histogram[bin3]++;
        h_Histogram[bin4]++;
    }

    // Process leftover elements, if there are any
    switch (nLeftOver)
    {
    case 0:
    default:
       // Do nothing
       break;

    case 1:
       // Process one element
       data1 = ((U32 *)h_Data)[idx+0u];
       mapper.Transform1( bin1, data1 );
       h_Histogram[bin1]++;
       break;

    case 2:
       // Process two elements
       data1 = ((U32 *)h_Data)[idx+0u];
       data2 = ((U32 *)h_Data)[idx+1u];
       mapper.Transform2( bin1, bin2, data1, data2 );
       h_Histogram[bin1]++;
       h_Histogram[bin2]++;
       break;

    case 3:
       // Process three elements
       data1 = ((U32 *)h_Data)[idx+0u];
       data2 = ((U32 *)h_Data)[idx+1u];
       data3 = ((U32 *)h_Data)[idx+2u];
       mapper.Transform3( bin1, bin2, bin3, data1, data2, data3 );
       h_Histogram[bin1]++;
       h_Histogram[bin2]++;
       h_Histogram[bin3]++;
       break;
    }
    
    // Cleanup mapping object
    mapper.Finish();

#if 0
    //-----
    // Print out CPU histogram counts
    //-----

    U32 total = 0u;
    for (U32 i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
    {
       U32 count = h_Histogram[i];
       total += count;
       fprintf( stdout, "Count[%u] = %u\n", i, count );
    }
    fprintf( stdout, "\nTotal = %u\n", total );
#endif
}


/*-----------------------------
  Name: CPU_histogramGenFloat_B4
  Desc: Verify CPU version of generic histogram code for FLOAT data
-----------------------------*/

extern "C" void CPU_histogramGenFloat_B4
(
    uint *h_Histogram,
    void *h_Data,
    uint byteCount,
    float minVal,
    float maxVal,
    uint nBins
)
{
    U32 floatCount = (U32)byteCount / 4u; // 4 bytes per FLOAT
    U32 testMod4 = (U32)byteCount % 4u;   // Make sure we are FLOAT aligned
    assert( (sizeof(U32) == 4u) && (testMod4 == 0u) );

    U32 nRows     = (floatCount/4u)*4u;  // Process 4 values at a time
    U32 nLeftOver = floatCount % 4u;     // Process any leftover FLOATS [0..3]

    //-----
    // Zero out histogram array
    //-----
    
    for (U32 i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
    {
        h_Histogram[i] = 0u;
    }


    //-----
    // Bin all data elements
    //-----

    // Initialize Mapping Object
    typedef MapToBin
            < 
               F32,     // Value type
               F32,     // Conversion type
               U32,     // Bin type
               1u,      // Formula #1: B=(A-Mu)*Alpha; where Mu = (min-0.5); and Alpha = n/(max-min+1)
               1u,      // Range check values < min
               1u       // Range check values > max
            > MapperF32;
    MapperF32 mapper;
    mapper.Initiate( (F32)minVal, (F32)maxVal, (U32)nBins );

    U32 bin1, bin2, bin3, bin4;
    F32 data1, data2, data3, data4;

    // Bin each element
    U32 idx;
    for (idx = 0u; idx < nRows; idx += 4u)
    {
        data1 = ((F32 *)h_Data)[idx+0u];
        data2 = ((F32 *)h_Data)[idx+1u];
        data3 = ((F32 *)h_Data)[idx+2u];
        data4 = ((F32 *)h_Data)[idx+3u];

        // Transform from 4 values to 4 bins
        mapper.Transform4( bin1, bin2, bin3, bin4,
                           data1, data2, data3, data4 );

        // Bin results
        h_Histogram[bin1]++;
        h_Histogram[bin2]++;
        h_Histogram[bin3]++;
        h_Histogram[bin4]++;
    }

    // Process leftover elements, if there are any
    switch (nLeftOver)
    {
    case 0:
    default:
       // Do nothing
       break;

    case 1:
       // Process one element
       data1 = ((F32 *)h_Data)[idx+0u];
       mapper.Transform1( bin1, data1 );
       h_Histogram[bin1]++;
       break;

    case 2:
       // Process two elements
       data1 = ((F32 *)h_Data)[idx+0u];
       data2 = ((F32 *)h_Data)[idx+1u];
       mapper.Transform2( bin1, bin2, data1, data2 );
       h_Histogram[bin1]++;
       h_Histogram[bin2]++;
       break;

    case 3:
       // Process three elements
       data1 = ((F32 *)h_Data)[idx+0u];
       data2 = ((F32 *)h_Data)[idx+1u];
       data3 = ((F32 *)h_Data)[idx+2u];
       mapper.Transform3( bin1, bin2, bin3, data1, data2, data3 );
       h_Histogram[bin1]++;
       h_Histogram[bin2]++;
       h_Histogram[bin3]++;
       break;
    }
    
    // Cleanup mapping object
    mapper.Finish();

#if 0
    //-----
    // Print out CPU histogram counts
    //-----

    U32 total = 0u;
    for (U32 i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
    {
       U32 count = h_Histogram[i];
       total += count;
       fprintf( stdout, "Count[%u] = %u\n", i, count );
    }
    fprintf( stdout, "\nTotal = %u\n", total );
#endif
}
