import numpy as np

from utils.matlabwrappers import *


def rlgr(R):
    #RLGR computes number of bits to code R with Adaptive Run Length Golomb Rice
    #
    # Copyright 8i Labs, Inc., 2017
    # This code is to be used solely for the purpose of developing the MPEG PCC standard.
    #
    # R = Nx1 array of integers to be coded
    #
    # bitStreamCount = number of bits needed to code R
    # bitStream = ceil(bitCount/8)x1 uint8 array containing bit stream

    # Length of input.
    N = len(R);

    # Constants.
    L = 4;
    U0 = 3;
    D0 = 1;
    U1 = 2;
    D1 = 1;
    quotientMax = 24;

    # Initialize state.
    k_P = 0;
    k_RP = 10*L;
    bitStreamCount = 0; # number of bits written to bitStream

    # Preprocess data from signed to unsigned.
    U = 2 * R;
    neg = (R < 0);
    Uneg = -U[neg];
    U[neg] = Uneg - 1;

    # Allocate space for bitstream if caller requests it.
    
    bitStream = np.zeros((2*N,1),np.uint8);
    byteStreamCount = 0;
    bitBuffer = np.zeros((1,1),np.uint64);
    bitBufferCount = 0;
    bitBufferCountMax = 64;

    # Process data one sample at a time (time consuming in Matlab).
    n = 1;
    while n <= N:
        
        k = floor(k_P / L);
        
        k_RP = min([k_RP,31*L]);
        k_R = floor(k_RP / L);
        pow2k_R = int(2**k_R);
        
        u = U[n-1]; # symbol to encode

        if k == 0: # no-run mode

            # Output GR code for symbol u.
            # bits = bits + gr(u,k_R);
            quotient = floor(u/pow2k_R); # number of 1s to write
            if quotient < quotientMax:
                # 'quotient' 1s + 0 + k_R-bit remainder
                bitStreamCount = bitStreamCount + (quotient + 1 + k_R);
            else:
                # 'quotientMax' 1s + 32-bit value of u
                bitStreamCount = bitStreamCount + quotientMax + 32;

            # Write to bitstream if caller requests it.

            if quotient < quotientMax:
                remainder = u - quotient*pow2k_R;
                bitPattern = bitor(bitshift(bitshift(uint64(1),quotient)-1,1+k_R),uint64(remainder));
                bitPatternCount = quotient + 1 + k_R;
            else:
                bitPattern = bitor(bitshift(bitshift(uint64(1),quotientMax)-1,32),uint64(u));
                bitPatternCount = quotientMax + 32;

            bitBuffer = bitor(bitshift(bitBuffer,bitPatternCount),bitPattern);
            bitBufferCount = bitBufferCount + bitPatternCount;
            if bitBufferCount > bitBufferCountMax:
                # Overflow error, not recoverable.
                bitStreamCount = -1;
                bitStream = uint8.empty;
                return;

            while bitBufferCount >= 8:
                bitBufferCount = bitBufferCount - 8;
                byteStreamCount = byteStreamCount + 1;
                bitStream[byteStreamCount-1] = bitand(bitshift(bitBuffer,-bitBufferCount),255);

            if bitStreamCount != 8*byteStreamCount + bitBufferCount:
                fprintf('error at n=%d\n',n);

            #print(bitand(bitshift(bitBuffer,-bitBufferCount),255))

            # Adapt k_R.
            p = floor(u/(2**k_R)); # number of probability halvings
            if p == 0:
                k_RP = max(0,k_RP - 2);
            elif p > 1:
                k_RP = k_RP + p + 1;


            # Adapt k.
            if u == 0:
                k_P = k_P + U0;
            else: # u > 0
                k_P = max(0,k_P - D0);
            
        else: # k > 0 # run mode

            m = bitshift(1,k); # m = 2**k = expected length of run of zeros

            # Parse off up to m symbols,
            # through first non-zero symbol,
            # counting number of zero symbols before it.
            zeroCount = 0;
            while u == 0:
                zeroCount = zeroCount + 1;
                if zeroCount >= m or n >= N:
                    break;

                n = n + 1;
                u = U[n-1];
            
            # At this point, either u>0 or (u=0 & (zeroCount>=m | n>=N).
            # That is, either u>0 or (u=0 & zeroCount>=m) or (u=0 & n>=N).
            if zeroCount == m:
                # Found a complete run of zeroCount = m zeros.
                # Output a 0.
                bitStreamCount = bitStreamCount + 1;

                # Write to bitstream if caller requests it.
                bitPattern = 0;
                bitPatternCount = 1;
                bitBuffer = bitor(bitshift(bitBuffer,bitPatternCount),bitPattern);
                bitBufferCount = bitBufferCount + bitPatternCount;
                while bitBufferCount >= 8:
                    bitBufferCount = bitBufferCount - 8;
                    byteStreamCount = byteStreamCount + 1;
                    bitStream[byteStreamCount-1] = bitand(bitshift(bitBuffer,-bitBufferCount),255);
                
                if bitStreamCount != 8*byteStreamCount + bitBufferCount:
                    fprintf('error at n=%d\n',n);
                
                # Adapt k.
                k_P = k_P + U1;

            else: # zeroCount < m, and either u>0 or (u=0 and n>=N)
                # Found a partial run of zeroCount < m zeros.
                if u > 0:
                    # Partial run ended normally with a non-zero symbol u.
                    # Output a 1 + length of partial run + GR code for non-zero symbol.
                    # bits = bits + 1 + k + gr(u-1,k_R);
                    quotient = floor((u-1)/pow2k_R); # number of 1s to write
                    if quotient < quotientMax:
                        bitStreamCount = bitStreamCount + 1 + k + (quotient + 1 + k_R);
                    else:
                        bitStreamCount = bitStreamCount + 1 + k + quotientMax + 32;
                    
                    # Write to bitstream if caller requests it.
                    bitPattern = bitor(m,zeroCount);
                    bitPatternCount = 1 + k;
                    bitBuffer = bitor(bitshift(bitBuffer,bitPatternCount),bitPattern);
                    bitBufferCount = bitBufferCount + bitPatternCount;
                    while bitBufferCount >= 8:
                        bitBufferCount = bitBufferCount - 8;
                        byteStreamCount = byteStreamCount + 1;
                        bitStream[byteStreamCount-1] = bitand(bitshift(bitBuffer,-bitBufferCount),255);
                    
                    if quotient < quotientMax:
                        remainder = (u-1) - quotient*pow2k_R;
                        bitPattern = bitor(bitshift(bitshift(uint64(1),quotient)-1,1+k_R),uint64(remainder));
                        bitPatternCount = quotient + 1 + k_R;
                    else:
                        bitPattern = bitor(bitshift(bitshift(uint64(1),quotientMax)-1,32),uint64(u));
                        bitPatternCount = quotientMax + 32;
                    
                    bitBuffer = bitor(bitshift(bitBuffer,bitPatternCount),bitPattern);
                    bitBufferCount = bitBufferCount + bitPatternCount;
                    if bitBufferCount > bitBufferCountMax:
                        # Overflow error, not recoverable.
                        bitStreamCount = -1;
                        bitStream = uint8.empty;
                        return;
                    
                    while bitBufferCount >= 8:
                        bitBufferCount = bitBufferCount - 8;
                        byteStreamCount = byteStreamCount + 1;
                        bitStream[byteStreamCount-1] = bitand(bitshift(bitBuffer,-bitBufferCount),255);
                    
                    if bitStreamCount != 8*byteStreamCount + bitBufferCount:
                        fprintf('error at n=%d\n',n);

                    # Adapt k_R.
                    p = floor((u-1)/(2**k_R)); # number of probability halvings
                    if p == 0:
                        k_RP = max(0,k_RP - 2);
                    elif p > 1:
                        k_RP = k_RP + p + 1;
                    

                    # Adapt k.
                    k_P = max(0,k_P - D1);
                    
                else: # u = 0 and n = N
                    # Partial run ended with a zero symbol, at end of sequence.
                    # Output a 0.  Leave it to decoder to know the number of symbols needed.
                    bitStreamCount = bitStreamCount + 1;

                    # Write to bitstream if caller requests it.

                    bitPattern = 0;
                    bitPatternCount = 1;
                    bitBuffer = bitor(bitshift(bitBuffer,bitPatternCount),bitPattern);
                    bitBufferCount = bitBufferCount + bitPatternCount;
                    while bitBufferCount >= 8:
                        bitBufferCount = bitBufferCount - 8;
                        byteStreamCount = byteStreamCount + 1;
                        bitStream[byteStreamCount-1] = bitand(bitshift(bitBuffer,-bitBufferCount),255);
                    
                    if bitStreamCount != 8*byteStreamCount + bitBufferCount:
                        fprintf('error at n=%d\n',n);

        n = n + 1;

    # Flush state to bitstream if caller requests it.
    
    if bitBufferCount > 0:
        byteStreamCount = byteStreamCount + 1;
        bitStream[byteStreamCount-1] = bitand(bitshift(bitBuffer,8-bitBufferCount),255);

    bitStream = bitStream[0:byteStreamCount];
    
    return bitStreamCount, bitStream


def irlgr(bitStream, N):
    # IRLGR decodes bitStream into integers using Adaptive Run Length Golomb Rice.
    #
    # Copyright 8i Labs, Inc., 2017
    # This code is to be used solely for the purpose of developing the MPEG PCC standard.
    #
    # bitStream = uint8 array to be decoded
    # R = Nx1 array of decoded signed integers

    # Constants.
    L = 4;
    U0 = 3;
    D0 = 1;
    U1 = 2;
    D1 = 1;
    quotientMax = 24;

    # Initialize state.
    k_P = 0;
    k_RP = 10*L;
    bitStreamCount = 0; # number of bits read from bitStream
    bitBufferCount = 0; # number of undecoded bits in bitBuffer
    bitBuffer = np.zeros((1,1),np.uint64); # initial value of bitBuffer
    bitStreamCountMax = 8 * len(bitStream);
    bitBufferCountMax = 64;

    # Allocate space for decoded unsigned integers.
    U = np.zeros((N,1));

    # Process data one sample at a time (time consuming in Matlab).
    n = 1;
    while n <= N:

        k = floor(k_P / L);
        k_RP = min([k_RP,31*L]);
        k_R = floor(k_RP / L);
        pow2k_R = 2**k_R;

        # Load up bitBuffer.
        while bitBufferCount <= bitBufferCountMax-8 and bitStreamCount <= bitStreamCountMax-8:
            bitBuffer = bitor(bitshift(bitBuffer,8),uint64(bitStream[int(bitStreamCount/8) + 1 - 1]));
            bitStreamCount = bitStreamCount + 8;
            bitBufferCount = bitBufferCount + 8;

        if k == 0: # no-run mode

            # Read floor(u/(2**k_R) + 1 + k_R bits for next symbol u>=0.

            # Read in quotient = floor(u/pow2k_R) 1s before a zero.
            quotient = 0;

            while quotient < quotientMax and bitget(bitBuffer,bitBufferCount) == 1:

                quotient = quotient + 1;
                bitBufferCount = bitBufferCount - 1;
            
            while bitBufferCount <= bitBufferCountMax-8 and bitStreamCount <= bitStreamCountMax-8:
                bitBuffer = bitor(bitshift(bitBuffer,8),uint64(bitStream[int(bitStreamCount/8) + 1 -1]));
                bitStreamCount = bitStreamCount + 8;
                bitBufferCount = bitBufferCount + 8;
            

            if quotient < quotientMax:
                # Read in the 0.
                bitBufferCount = bitBufferCount - 1;

                # Read in k_R bits, containing the remainder u - floor(u/pow2k_R).
                u = quotient*pow2k_R + bitand(bitshift(bitBuffer,k_R-bitBufferCount),pow2k_R-1);
                bitBufferCount = bitBufferCount - k_R;
            else:
                # Read in 32 bits, containing u.
                u = bitand(bitshift(bitBuffer,32-bitBufferCount),2**32-1);
                bitBufferCount = bitBufferCount - 32;

            # Output the decoded symbol u >= 0.
            U[n-1] = u;
            n = n + 1;

            # Adapt k_R.
            p = floor(double(u)/(2**k_R)); # number of probability halvings

            if p == 0:
                k_RP = max(0,k_RP - 2);
            elif p > 1:
                k_RP = k_RP + p + 1;

            # Adapt k.
            if u == 0:
                k_P = k_P + U0;
            else: # u > 0
                k_P = max(0,k_P - D0);
            
        else: # k > 0 # run mode

            m = bitshift(1,k); # m = 2**k = expected length of run of zeros

            # Read in next bit.
            if bitget(bitBuffer,bitBufferCount) == 0:
                bitBufferCount = bitBufferCount - 1;

                # Bit is 0, which means there is a complete run of m zeros.

                # Output the decoded zeros.
                while m > 0 and n <= N:
                    U[n-1] = 0;
                    n = n + 1;
                    m = m - 1;

                # Adapt k.
                k_P = k_P + U1;
            else: # bitvalue(bitBuffer,bitBufferCount) == 1
                bitBufferCount = bitBufferCount - 1;

                # Bit is 1, which means there is a partial run of zeroCount < m zeros.

                # First read in k bits to specify zeroCount.
                zeroCount = bitand(bitshift(bitBuffer,k-bitBufferCount),m-1);
                bitBufferCount = bitBufferCount - k;

                # Output the decoded zeros.
                while zeroCount > 0:
                    U[n-1] = 0;
                    n = n + 1;
                    zeroCount = zeroCount - 1;

                # Then read in floor((u-1)/(2**k_R) + 1 + k_R bits for next symbol u >= 1.

                # Read in quotient = floor((u-1)/pow2k_R) 1s before a zero.
                quotient = 0;
                while quotient < quotientMax and bitget(bitBuffer,bitBufferCount) == 1:
                    quotient = quotient + 1;
                    bitBufferCount = bitBufferCount - 1;
                
                while bitBufferCount <= bitBufferCountMax-8 and bitStreamCount <= bitStreamCountMax-8:
                    bitBuffer = bitor(bitshift(bitBuffer,8),uint64(bitStream[int(bitStreamCount/8) + 1 -1]));
                    bitStreamCount = bitStreamCount + 8;
                    bitBufferCount = bitBufferCount + 8;

                if quotient < quotientMax:
                    # Read in the 0.
                    bitBufferCount = bitBufferCount - 1;

                    # Read in k_R bits, containing the remainder (u-1) - floor((u-1)/pow2k_R).
                    u = 1 + quotient*pow2k_R + bitand(bitshift(bitBuffer,k_R-bitBufferCount),pow2k_R-1);
                    bitBufferCount = bitBufferCount - k_R;
                else:
                    # Read in 32 bits, containing u.
                    u = bitand(bitshift(bitBuffer,32-bitBufferCount),2**32-1);
                    bitBufferCount = bitBufferCount - 32;

                # Output the decoded symbol u >= 1.
                U[n-1] = u;
                n = n + 1;

                # Adapt k_R.
                p = floor(double(u-1)/(2**k_R)); # number of probability halvings
                if p == 0:
                    k_RP = max(0,k_RP - 2);
                elif p > 1:
                    k_RP = k_RP + p + 1;

                # Adapt k.
                k_P = max(0,k_P - D1);

    # Postprocess data from unsigned to signed.
    R = np.zeros((N,1));
    even = (np.mod(U,2)==0);
    R[even] = U[even] / 2;
    R[np.invert(even)] = -(U[np.invert(even)]+1) / 2;
                                     
    return R
