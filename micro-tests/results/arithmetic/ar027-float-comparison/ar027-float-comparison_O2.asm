
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar027-float-comparison/ar027-float-comparison_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_float_compareff>:
100000360: 1e212000    	fcmp	s0, s1
100000364: 5a9f53e8    	csetm	w8, mi
100000368: 1a9fd500    	csinc	w0, w8, wzr, le
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800020    	mov	w0, #0x1                ; =1
100000374: d65f03c0    	ret
