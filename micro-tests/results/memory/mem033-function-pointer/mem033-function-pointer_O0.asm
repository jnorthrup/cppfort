
/Users/jim/work/cppfort/micro-tests/results/memory/mem033-function-pointer/mem033-function-pointer_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z3addii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b9400be9    	ldr	w9, [sp, #0x8]
100000374: 0b090100    	add	w0, w8, w9
100000378: 910043ff    	add	sp, sp, #0x10
10000037c: d65f03c0    	ret

0000000100000380 <__Z21test_function_pointerv>:
100000380: d10083ff    	sub	sp, sp, #0x20
100000384: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000388: 910043fd    	add	x29, sp, #0x10
10000038c: 90000008    	adrp	x8, 0x100000000
100000390: 910d8108    	add	x8, x8, #0x360
100000394: f90007e8    	str	x8, [sp, #0x8]
100000398: f94007e8    	ldr	x8, [sp, #0x8]
10000039c: 52800060    	mov	w0, #0x3                ; =3
1000003a0: 52800081    	mov	w1, #0x4                ; =4
1000003a4: d63f0100    	blr	x8
1000003a8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003ac: 910083ff    	add	sp, sp, #0x20
1000003b0: d65f03c0    	ret

00000001000003b4 <_main>:
1000003b4: d10083ff    	sub	sp, sp, #0x20
1000003b8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003bc: 910043fd    	add	x29, sp, #0x10
1000003c0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c4: 97ffffef    	bl	0x100000380 <__Z21test_function_pointerv>
1000003c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003cc: 910083ff    	add	sp, sp, #0x20
1000003d0: d65f03c0    	ret
