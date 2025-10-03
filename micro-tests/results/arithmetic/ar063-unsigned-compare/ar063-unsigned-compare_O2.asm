
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar063-unsigned-compare/ar063-unsigned-compare_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z21test_unsigned_comparejj>:
100000360: 6b01001f    	cmp	w0, w1
100000364: 1a9f27e0    	cset	w0, lo
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 52800020    	mov	w0, #0x1                ; =1
100000370: d65f03c0    	ret
