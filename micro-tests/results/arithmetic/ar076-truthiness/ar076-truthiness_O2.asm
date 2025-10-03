
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar076-truthiness/ar076-truthiness_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_truthinessi>:
100000360: 7100001f    	cmp	w0, #0x0
100000364: 1a9f07e0    	cset	w0, ne
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 52800020    	mov	w0, #0x1                ; =1
100000370: d65f03c0    	ret
