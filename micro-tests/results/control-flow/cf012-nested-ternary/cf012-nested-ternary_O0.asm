
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf012-nested-ternary/cf012-nested-ternary_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_nested_ternaryi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe8    	ldr	w8, [sp, #0xc]
10000036c: 71000108    	subs	w8, w8, #0x0
100000370: 5400010d    	b.le	0x100000390 <__Z19test_nested_ternaryi+0x30>
100000374: 14000001    	b	0x100000378 <__Z19test_nested_ternaryi+0x18>
100000378: b9400fe9    	ldr	w9, [sp, #0xc]
10000037c: 52800048    	mov	w8, #0x2                ; =2
100000380: 71002929    	subs	w9, w9, #0xa
100000384: 1a9fc508    	csinc	w8, w8, wzr, gt
100000388: b9000be8    	str	w8, [sp, #0x8]
10000038c: 14000004    	b	0x10000039c <__Z19test_nested_ternaryi+0x3c>
100000390: 52800008    	mov	w8, #0x0                ; =0
100000394: b9000be8    	str	w8, [sp, #0x8]
100000398: 14000001    	b	0x10000039c <__Z19test_nested_ternaryi+0x3c>
10000039c: b9400be0    	ldr	w0, [sp, #0x8]
1000003a0: 910043ff    	add	sp, sp, #0x10
1000003a4: d65f03c0    	ret

00000001000003a8 <_main>:
1000003a8: d10083ff    	sub	sp, sp, #0x20
1000003ac: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b0: 910043fd    	add	x29, sp, #0x10
1000003b4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003b8: 528000a0    	mov	w0, #0x5                ; =5
1000003bc: 97ffffe9    	bl	0x100000360 <__Z19test_nested_ternaryi>
1000003c0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003c4: 910083ff    	add	sp, sp, #0x20
1000003c8: d65f03c0    	ret
