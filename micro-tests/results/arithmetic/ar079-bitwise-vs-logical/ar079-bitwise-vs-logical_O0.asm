
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar079-bitwise-vs-logical/ar079-bitwise-vs-logical_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_bitwise_vs_logicalii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b9400be9    	ldr	w9, [sp, #0x8]
100000374: 0a090108    	and	w8, w8, w9
100000378: b90003e8    	str	w8, [sp]
10000037c: b9400fe8    	ldr	w8, [sp, #0xc]
100000380: 52800009    	mov	w9, #0x0                ; =0
100000384: b90007e9    	str	w9, [sp, #0x4]
100000388: 340000e8    	cbz	w8, 0x1000003a4 <__Z23test_bitwise_vs_logicalii+0x44>
10000038c: 14000001    	b	0x100000390 <__Z23test_bitwise_vs_logicalii+0x30>
100000390: b9400be8    	ldr	w8, [sp, #0x8]
100000394: 71000108    	subs	w8, w8, #0x0
100000398: 1a9f07e8    	cset	w8, ne
10000039c: b90007e8    	str	w8, [sp, #0x4]
1000003a0: 14000001    	b	0x1000003a4 <__Z23test_bitwise_vs_logicalii+0x44>
1000003a4: b94003e8    	ldr	w8, [sp]
1000003a8: b94007e9    	ldr	w9, [sp, #0x4]
1000003ac: 12000129    	and	w9, w9, #0x1
1000003b0: 6b090108    	subs	w8, w8, w9
1000003b4: 1a9f07e0    	cset	w0, ne
1000003b8: 910043ff    	add	sp, sp, #0x10
1000003bc: d65f03c0    	ret

00000001000003c0 <_main>:
1000003c0: d10083ff    	sub	sp, sp, #0x20
1000003c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c8: 910043fd    	add	x29, sp, #0x10
1000003cc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d0: 52800040    	mov	w0, #0x2                ; =2
1000003d4: 52800061    	mov	w1, #0x3                ; =3
1000003d8: 97ffffe2    	bl	0x100000360 <__Z23test_bitwise_vs_logicalii>
1000003dc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e0: 910083ff    	add	sp, sp, #0x20
1000003e4: d65f03c0    	ret
