
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar027-float-comparison/ar027-float-comparison_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_float_compareff>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: bd000be0    	str	s0, [sp, #0x8]
100000368: bd0007e1    	str	s1, [sp, #0x4]
10000036c: bd400be0    	ldr	s0, [sp, #0x8]
100000370: bd4007e1    	ldr	s1, [sp, #0x4]
100000374: 1e212000    	fcmp	s0, s1
100000378: 540000ad    	b.le	0x10000038c <__Z18test_float_compareff+0x2c>
10000037c: 14000001    	b	0x100000380 <__Z18test_float_compareff+0x20>
100000380: 52800028    	mov	w8, #0x1                ; =1
100000384: b9000fe8    	str	w8, [sp, #0xc]
100000388: 1400000b    	b	0x1000003b4 <__Z18test_float_compareff+0x54>
10000038c: bd400be0    	ldr	s0, [sp, #0x8]
100000390: bd4007e1    	ldr	s1, [sp, #0x4]
100000394: 1e212000    	fcmp	s0, s1
100000398: 540000a5    	b.pl	0x1000003ac <__Z18test_float_compareff+0x4c>
10000039c: 14000001    	b	0x1000003a0 <__Z18test_float_compareff+0x40>
1000003a0: 12800008    	mov	w8, #-0x1               ; =-1
1000003a4: b9000fe8    	str	w8, [sp, #0xc]
1000003a8: 14000003    	b	0x1000003b4 <__Z18test_float_compareff+0x54>
1000003ac: b9000fff    	str	wzr, [sp, #0xc]
1000003b0: 14000001    	b	0x1000003b4 <__Z18test_float_compareff+0x54>
1000003b4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003b8: 910043ff    	add	sp, sp, #0x10
1000003bc: d65f03c0    	ret

00000001000003c0 <_main>:
1000003c0: d10083ff    	sub	sp, sp, #0x20
1000003c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c8: 910043fd    	add	x29, sp, #0x10
1000003cc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d0: 1e219000    	fmov	s0, #3.50000000
1000003d4: 1e209001    	fmov	s1, #2.50000000
1000003d8: 97ffffe2    	bl	0x100000360 <__Z18test_float_compareff>
1000003dc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e0: 910083ff    	add	sp, sp, #0x20
1000003e4: d65f03c0    	ret
