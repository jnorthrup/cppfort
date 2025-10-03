
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar078-logical-precedence/ar078-logical-precedence_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_logical_precedencebbb>:
100000360: 0a020028    	and	w8, w1, w2
100000364: 2a000100    	orr	w0, w8, w0
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 52800020    	mov	w0, #0x1                ; =1
100000370: d65f03c0    	ret
