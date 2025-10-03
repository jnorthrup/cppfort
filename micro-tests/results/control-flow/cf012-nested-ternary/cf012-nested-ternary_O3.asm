
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf012-nested-ternary/cf012-nested-ternary_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_nested_ternaryi>:
100000360: 7100281f    	cmp	w0, #0xa
100000364: 52800028    	mov	w8, #0x1                ; =1
100000368: 1a889508    	cinc	w8, w8, hi
10000036c: 7100001f    	cmp	w0, #0x0
100000370: 1a9fc100    	csel	w0, w8, wzr, gt
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: 52800020    	mov	w0, #0x1                ; =1
10000037c: d65f03c0    	ret
