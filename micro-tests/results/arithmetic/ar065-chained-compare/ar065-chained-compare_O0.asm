
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar065-chained-compare/ar065-chained-compare_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_chained_compareiii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b90007e2    	str	w2, [sp, #0x4]
100000370: b9400fe9    	ldr	w9, [sp, #0xc]
100000374: b9400bea    	ldr	w10, [sp, #0x8]
100000378: 52800008    	mov	w8, #0x0                ; =0
10000037c: 6b0a0129    	subs	w9, w9, w10
100000380: b90003e8    	str	w8, [sp]
100000384: 5400010a    	b.ge	0x1000003a4 <__Z20test_chained_compareiii+0x44>
100000388: 14000001    	b	0x10000038c <__Z20test_chained_compareiii+0x2c>
10000038c: b9400be8    	ldr	w8, [sp, #0x8]
100000390: b94007e9    	ldr	w9, [sp, #0x4]
100000394: 6b090108    	subs	w8, w8, w9
100000398: 1a9fa7e8    	cset	w8, lt
10000039c: b90003e8    	str	w8, [sp]
1000003a0: 14000001    	b	0x1000003a4 <__Z20test_chained_compareiii+0x44>
1000003a4: b94003e8    	ldr	w8, [sp]
1000003a8: 12000100    	and	w0, w8, #0x1
1000003ac: 910043ff    	add	sp, sp, #0x10
1000003b0: d65f03c0    	ret

00000001000003b4 <_main>:
1000003b4: d10083ff    	sub	sp, sp, #0x20
1000003b8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003bc: 910043fd    	add	x29, sp, #0x10
1000003c0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c4: 52800020    	mov	w0, #0x1                ; =1
1000003c8: 528000a1    	mov	w1, #0x5                ; =5
1000003cc: 52800142    	mov	w2, #0xa                ; =10
1000003d0: 97ffffe4    	bl	0x100000360 <__Z20test_chained_compareiii>
1000003d4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d8: 910083ff    	add	sp, sp, #0x20
1000003dc: d65f03c0    	ret
