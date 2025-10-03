
/Users/jim/work/cppfort/micro-tests/results/memory/mem085-reference-vs-pointer/mem085-reference-vs-pointer_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_ref_vs_ptrv>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: 910073e8    	add	x8, sp, #0x1c
100000368: 52800549    	mov	w9, #0x2a               ; =42
10000036c: b9001fe9    	str	w9, [sp, #0x1c]
100000370: aa0803e9    	mov	x9, x8
100000374: f9000be9    	str	x9, [sp, #0x10]
100000378: f90007e8    	str	x8, [sp, #0x8]
10000037c: f9400be8    	ldr	x8, [sp, #0x10]
100000380: b9400109    	ldr	w9, [x8]
100000384: f94007e8    	ldr	x8, [sp, #0x8]
100000388: b940010a    	ldr	w10, [x8]
10000038c: 52800008    	mov	w8, #0x0                ; =0
100000390: 6b0a0129    	subs	w9, w9, w10
100000394: 1a9f1500    	csinc	w0, w8, wzr, ne
100000398: 910083ff    	add	sp, sp, #0x20
10000039c: d65f03c0    	ret

00000001000003a0 <_main>:
1000003a0: d10083ff    	sub	sp, sp, #0x20
1000003a4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a8: 910043fd    	add	x29, sp, #0x10
1000003ac: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003b0: 97ffffec    	bl	0x100000360 <__Z15test_ref_vs_ptrv>
1000003b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b8: 910083ff    	add	sp, sp, #0x20
1000003bc: d65f03c0    	ret
