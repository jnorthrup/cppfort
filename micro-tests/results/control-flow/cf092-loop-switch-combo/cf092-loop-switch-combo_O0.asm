
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf092-loop-switch-combo/cf092-loop-switch-combo_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_loop_switchv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z16test_loop_switchv+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71002908    	subs	w8, w8, #0xa
100000378: 5400046a    	b.ge	0x100000404 <__Z16test_loop_switchv+0xa4>
10000037c: 14000001    	b	0x100000380 <__Z16test_loop_switchv+0x20>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 5280006a    	mov	w10, #0x3               ; =3
100000388: 1aca0d09    	sdiv	w9, w8, w10
10000038c: 1b0a7d29    	mul	w9, w9, w10
100000390: 6b090108    	subs	w8, w8, w9
100000394: b90007e8    	str	w8, [sp, #0x4]
100000398: 34000148    	cbz	w8, 0x1000003c0 <__Z16test_loop_switchv+0x60>
10000039c: 14000001    	b	0x1000003a0 <__Z16test_loop_switchv+0x40>
1000003a0: b94007e8    	ldr	w8, [sp, #0x4]
1000003a4: 71000508    	subs	w8, w8, #0x1
1000003a8: 54000140    	b.eq	0x1000003d0 <__Z16test_loop_switchv+0x70>
1000003ac: 14000001    	b	0x1000003b0 <__Z16test_loop_switchv+0x50>
1000003b0: b94007e8    	ldr	w8, [sp, #0x4]
1000003b4: 71000908    	subs	w8, w8, #0x2
1000003b8: 54000140    	b.eq	0x1000003e0 <__Z16test_loop_switchv+0x80>
1000003bc: 1400000d    	b	0x1000003f0 <__Z16test_loop_switchv+0x90>
1000003c0: b9400fe8    	ldr	w8, [sp, #0xc]
1000003c4: 11000508    	add	w8, w8, #0x1
1000003c8: b9000fe8    	str	w8, [sp, #0xc]
1000003cc: 14000009    	b	0x1000003f0 <__Z16test_loop_switchv+0x90>
1000003d0: b9400fe8    	ldr	w8, [sp, #0xc]
1000003d4: 11000908    	add	w8, w8, #0x2
1000003d8: b9000fe8    	str	w8, [sp, #0xc]
1000003dc: 14000005    	b	0x1000003f0 <__Z16test_loop_switchv+0x90>
1000003e0: b9400fe8    	ldr	w8, [sp, #0xc]
1000003e4: 11000d08    	add	w8, w8, #0x3
1000003e8: b9000fe8    	str	w8, [sp, #0xc]
1000003ec: 14000001    	b	0x1000003f0 <__Z16test_loop_switchv+0x90>
1000003f0: 14000001    	b	0x1000003f4 <__Z16test_loop_switchv+0x94>
1000003f4: b9400be8    	ldr	w8, [sp, #0x8]
1000003f8: 11000508    	add	w8, w8, #0x1
1000003fc: b9000be8    	str	w8, [sp, #0x8]
100000400: 17ffffdc    	b	0x100000370 <__Z16test_loop_switchv+0x10>
100000404: b9400fe0    	ldr	w0, [sp, #0xc]
100000408: 910043ff    	add	sp, sp, #0x10
10000040c: d65f03c0    	ret

0000000100000410 <_main>:
100000410: d10083ff    	sub	sp, sp, #0x20
100000414: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000418: 910043fd    	add	x29, sp, #0x10
10000041c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000420: 97ffffd0    	bl	0x100000360 <__Z16test_loop_switchv>
100000424: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000428: 910083ff    	add	sp, sp, #0x20
10000042c: d65f03c0    	ret
