
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar078-logical-precedence/ar078-logical-precedence_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_logical_precedencebbb>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 39003fe0    	strb	w0, [sp, #0xf]
100000368: 39003be1    	strb	w1, [sp, #0xe]
10000036c: 390037e2    	strb	w2, [sp, #0xd]
100000370: 39403fe8    	ldrb	w8, [sp, #0xf]
100000374: 52800029    	mov	w9, #0x1                ; =1
100000378: b9000be9    	str	w9, [sp, #0x8]
10000037c: 370001a8    	tbnz	w8, #0x0, 0x1000003b0 <__Z23test_logical_precedencebbb+0x50>
100000380: 14000001    	b	0x100000384 <__Z23test_logical_precedencebbb+0x24>
100000384: 39403be8    	ldrb	w8, [sp, #0xe]
100000388: 52800009    	mov	w9, #0x0                ; =0
10000038c: b90007e9    	str	w9, [sp, #0x4]
100000390: 360000a8    	tbz	w8, #0x0, 0x1000003a4 <__Z23test_logical_precedencebbb+0x44>
100000394: 14000001    	b	0x100000398 <__Z23test_logical_precedencebbb+0x38>
100000398: 394037e8    	ldrb	w8, [sp, #0xd]
10000039c: b90007e8    	str	w8, [sp, #0x4]
1000003a0: 14000001    	b	0x1000003a4 <__Z23test_logical_precedencebbb+0x44>
1000003a4: b94007e8    	ldr	w8, [sp, #0x4]
1000003a8: b9000be8    	str	w8, [sp, #0x8]
1000003ac: 14000001    	b	0x1000003b0 <__Z23test_logical_precedencebbb+0x50>
1000003b0: b9400be8    	ldr	w8, [sp, #0x8]
1000003b4: 12000100    	and	w0, w8, #0x1
1000003b8: 910043ff    	add	sp, sp, #0x10
1000003bc: d65f03c0    	ret

00000001000003c0 <_main>:
1000003c0: d10083ff    	sub	sp, sp, #0x20
1000003c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c8: 910043fd    	add	x29, sp, #0x10
1000003cc: 52800009    	mov	w9, #0x0                ; =0
1000003d0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d4: 52800028    	mov	w8, #0x1                ; =1
1000003d8: 12000120    	and	w0, w9, #0x1
1000003dc: 12000101    	and	w1, w8, #0x1
1000003e0: 12000102    	and	w2, w8, #0x1
1000003e4: 97ffffdf    	bl	0x100000360 <__Z23test_logical_precedencebbb>
1000003e8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003ec: 910083ff    	add	sp, sp, #0x20
1000003f0: d65f03c0    	ret
