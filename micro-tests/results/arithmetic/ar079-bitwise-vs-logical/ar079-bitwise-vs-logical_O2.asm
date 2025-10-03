
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar079-bitwise-vs-logical/ar079-bitwise-vs-logical_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_bitwise_vs_logicalii>:
100000360: 0a000028    	and	w8, w1, w0
100000364: 7100001f    	cmp	w0, #0x0
100000368: 7a401824    	ccmp	w1, #0x0, #0x4, ne
10000036c: 1a9f07e9    	cset	w9, ne
100000370: 6b09011f    	cmp	w8, w9
100000374: 1a9f07e0    	cset	w0, ne
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: 52800020    	mov	w0, #0x1                ; =1
100000380: d65f03c0    	ret
