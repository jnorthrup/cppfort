
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf096-deeply-nested/cf096-deeply-nested_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_deeply_nestedi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: 71000108    	subs	w8, w8, #0x0
100000370: 5400034d    	b.le	0x1000003d8 <__Z18test_deeply_nestedi+0x78>
100000374: 14000001    	b	0x100000378 <__Z18test_deeply_nestedi+0x18>
100000378: b9400be8    	ldr	w8, [sp, #0x8]
10000037c: 71002908    	subs	w8, w8, #0xa
100000380: 5400026d    	b.le	0x1000003cc <__Z18test_deeply_nestedi+0x6c>
100000384: 14000001    	b	0x100000388 <__Z18test_deeply_nestedi+0x28>
100000388: b9400be8    	ldr	w8, [sp, #0x8]
10000038c: 71005108    	subs	w8, w8, #0x14
100000390: 5400018d    	b.le	0x1000003c0 <__Z18test_deeply_nestedi+0x60>
100000394: 14000001    	b	0x100000398 <__Z18test_deeply_nestedi+0x38>
100000398: b9400be8    	ldr	w8, [sp, #0x8]
10000039c: 71007908    	subs	w8, w8, #0x1e
1000003a0: 540000ad    	b.le	0x1000003b4 <__Z18test_deeply_nestedi+0x54>
1000003a4: 14000001    	b	0x1000003a8 <__Z18test_deeply_nestedi+0x48>
1000003a8: 52800088    	mov	w8, #0x4                ; =4
1000003ac: b9000fe8    	str	w8, [sp, #0xc]
1000003b0: 1400000c    	b	0x1000003e0 <__Z18test_deeply_nestedi+0x80>
1000003b4: 52800068    	mov	w8, #0x3                ; =3
1000003b8: b9000fe8    	str	w8, [sp, #0xc]
1000003bc: 14000009    	b	0x1000003e0 <__Z18test_deeply_nestedi+0x80>
1000003c0: 52800048    	mov	w8, #0x2                ; =2
1000003c4: b9000fe8    	str	w8, [sp, #0xc]
1000003c8: 14000006    	b	0x1000003e0 <__Z18test_deeply_nestedi+0x80>
1000003cc: 52800028    	mov	w8, #0x1                ; =1
1000003d0: b9000fe8    	str	w8, [sp, #0xc]
1000003d4: 14000003    	b	0x1000003e0 <__Z18test_deeply_nestedi+0x80>
1000003d8: b9000fff    	str	wzr, [sp, #0xc]
1000003dc: 14000001    	b	0x1000003e0 <__Z18test_deeply_nestedi+0x80>
1000003e0: b9400fe0    	ldr	w0, [sp, #0xc]
1000003e4: 910043ff    	add	sp, sp, #0x10
1000003e8: d65f03c0    	ret

00000001000003ec <_main>:
1000003ec: d10083ff    	sub	sp, sp, #0x20
1000003f0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003f4: 910043fd    	add	x29, sp, #0x10
1000003f8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003fc: 52800320    	mov	w0, #0x19               ; =25
100000400: 97ffffd8    	bl	0x100000360 <__Z18test_deeply_nestedi>
100000404: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000408: 910083ff    	add	sp, sp, #0x20
10000040c: d65f03c0    	ret
