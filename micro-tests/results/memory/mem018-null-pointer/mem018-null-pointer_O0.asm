
/Users/jim/work/cppfort/micro-tests/results/memory/mem018-null-pointer/mem018-null-pointer_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_null_pointerv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: f90007ff    	str	xzr, [sp, #0x8]
100000368: f94007e9    	ldr	x9, [sp, #0x8]
10000036c: 52800008    	mov	w8, #0x0                ; =0
100000370: f1000129    	subs	x9, x9, #0x0
100000374: 1a9f1500    	csinc	w0, w8, wzr, ne
100000378: 910043ff    	add	sp, sp, #0x10
10000037c: d65f03c0    	ret

0000000100000380 <_main>:
100000380: d10083ff    	sub	sp, sp, #0x20
100000384: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000388: 910043fd    	add	x29, sp, #0x10
10000038c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000390: 97fffff4    	bl	0x100000360 <__Z17test_null_pointerv>
100000394: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000398: 910083ff    	add	sp, sp, #0x20
10000039c: d65f03c0    	ret
