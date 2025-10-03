
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf086-short-circuit-side-effects/cf086-short-circuit-side-effects_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_side_effectsi>:
100000360: 52800148    	mov	w8, #0xa                ; =10
100000364: 7100001f    	cmp	w0, #0x0
100000368: 1a9fc100    	csel	w0, w8, wzr, gt
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800140    	mov	w0, #0xa                ; =10
100000374: d65f03c0    	ret
