
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar020-complex-expr/ar020-complex-expr_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z12test_complexiiii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b90007e2    	str	w2, [sp, #0x4]
100000370: b90003e3    	str	w3, [sp]
100000374: b9400fe8    	ldr	w8, [sp, #0xc]
100000378: b9400be9    	ldr	w9, [sp, #0x8]
10000037c: 0b090108    	add	w8, w8, w9
100000380: b94007e9    	ldr	w9, [sp, #0x4]
100000384: b94003ea    	ldr	w10, [sp]
100000388: 6b0a0129    	subs	w9, w9, w10
10000038c: 1b097d08    	mul	w8, w8, w9
100000390: 52800049    	mov	w9, #0x2                ; =2
100000394: 1ac90d00    	sdiv	w0, w8, w9
100000398: 910043ff    	add	sp, sp, #0x10
10000039c: d65f03c0    	ret

00000001000003a0 <_main>:
1000003a0: d10083ff    	sub	sp, sp, #0x20
1000003a4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a8: 910043fd    	add	x29, sp, #0x10
1000003ac: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003b0: 52800140    	mov	w0, #0xa                ; =10
1000003b4: 528000a1    	mov	w1, #0x5                ; =5
1000003b8: 52800282    	mov	w2, #0x14               ; =20
1000003bc: 52800083    	mov	w3, #0x4                ; =4
1000003c0: 97ffffe8    	bl	0x100000360 <__Z12test_complexiiii>
1000003c4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003c8: 910083ff    	add	sp, sp, #0x20
1000003cc: d65f03c0    	ret
