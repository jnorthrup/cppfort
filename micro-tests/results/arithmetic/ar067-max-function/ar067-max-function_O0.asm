
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar067-max-function/ar067-max-function_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_maxii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b9400be9    	ldr	w9, [sp, #0x8]
100000374: 6b090108    	subs	w8, w8, w9
100000378: 540000ad    	b.le	0x10000038c <__Z8test_maxii+0x2c>
10000037c: 14000001    	b	0x100000380 <__Z8test_maxii+0x20>
100000380: b9400fe8    	ldr	w8, [sp, #0xc]
100000384: b90007e8    	str	w8, [sp, #0x4]
100000388: 14000004    	b	0x100000398 <__Z8test_maxii+0x38>
10000038c: b9400be8    	ldr	w8, [sp, #0x8]
100000390: b90007e8    	str	w8, [sp, #0x4]
100000394: 14000001    	b	0x100000398 <__Z8test_maxii+0x38>
100000398: b94007e0    	ldr	w0, [sp, #0x4]
10000039c: 910043ff    	add	sp, sp, #0x10
1000003a0: d65f03c0    	ret

00000001000003a4 <_main>:
1000003a4: d10083ff    	sub	sp, sp, #0x20
1000003a8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003ac: 910043fd    	add	x29, sp, #0x10
1000003b0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003b4: 528000a0    	mov	w0, #0x5                ; =5
1000003b8: 52800061    	mov	w1, #0x3                ; =3
1000003bc: 97ffffe9    	bl	0x100000360 <__Z8test_maxii>
1000003c0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003c4: 910083ff    	add	sp, sp, #0x20
1000003c8: d65f03c0    	ret
