
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar032-long-double/ar032-long-double_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_long_doubleee>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: fd0007e0    	str	d0, [sp, #0x8]
100000368: fd0003e1    	str	d1, [sp]
10000036c: fd4007e0    	ldr	d0, [sp, #0x8]
100000370: fd4003e1    	ldr	d1, [sp]
100000374: 1e610800    	fmul	d0, d0, d1
100000378: 910043ff    	add	sp, sp, #0x10
10000037c: d65f03c0    	ret

0000000100000380 <_main>:
100000380: d10083ff    	sub	sp, sp, #0x20
100000384: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000388: 910043fd    	add	x29, sp, #0x10
10000038c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000390: 1e611000    	fmov	d0, #3.00000000
100000394: 1e621001    	fmov	d1, #4.00000000
100000398: 97fffff2    	bl	0x100000360 <__Z16test_long_doubleee>
10000039c: 1e780000    	fcvtzs	w0, d0
1000003a0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a4: 910083ff    	add	sp, sp, #0x20
1000003a8: d65f03c0    	ret
