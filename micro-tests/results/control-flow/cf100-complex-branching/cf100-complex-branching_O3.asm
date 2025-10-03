
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf100-complex-branching/cf100-complex-branching_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z22test_complex_branchingiii>:
100000360: 6b02003f    	cmp	w1, w2
100000364: 1a9fc028    	csel	w8, w1, wzr, gt
100000368: 0b020108    	add	w8, w8, w2
10000036c: 0b000029    	add	w9, w1, w0
100000370: 0b02012a    	add	w10, w9, w2
100000374: 6b02001f    	cmp	w0, w2
100000378: 1a89d108    	csel	w8, w8, w9, le
10000037c: 1a9fc009    	csel	w9, w0, wzr, gt
100000380: 0b020129    	add	w9, w9, w2
100000384: 6b02003f    	cmp	w1, w2
100000388: 1a8ad129    	csel	w9, w9, w10, le
10000038c: 6b01001f    	cmp	w0, w1
100000390: 1a89d100    	csel	w0, w8, w9, le
100000394: d65f03c0    	ret

0000000100000398 <_main>:
100000398: 528001e0    	mov	w0, #0xf                ; =15
10000039c: d65f03c0    	ret
