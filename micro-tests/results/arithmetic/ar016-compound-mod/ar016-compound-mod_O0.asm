
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar016-compound-mod/ar016-compound-mod_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_compound_modii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b90007e1    	str	w1, [sp, #0x4]
10000036c: b94007e8    	ldr	w8, [sp, #0x4]
100000370: 35000088    	cbnz	w8, 0x100000380 <__Z17test_compound_modii+0x20>
100000374: 14000001    	b	0x100000378 <__Z17test_compound_modii+0x18>
100000378: b9000fff    	str	wzr, [sp, #0xc]
10000037c: 1400000a    	b	0x1000003a4 <__Z17test_compound_modii+0x44>
100000380: b94007ea    	ldr	w10, [sp, #0x4]
100000384: b9400be8    	ldr	w8, [sp, #0x8]
100000388: 1aca0d09    	sdiv	w9, w8, w10
10000038c: 1b0a7d29    	mul	w9, w9, w10
100000390: 6b090108    	subs	w8, w8, w9
100000394: b9000be8    	str	w8, [sp, #0x8]
100000398: b9400be8    	ldr	w8, [sp, #0x8]
10000039c: b9000fe8    	str	w8, [sp, #0xc]
1000003a0: 14000001    	b	0x1000003a4 <__Z17test_compound_modii+0x44>
1000003a4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003a8: 910043ff    	add	sp, sp, #0x10
1000003ac: d65f03c0    	ret

00000001000003b0 <_main>:
1000003b0: d10083ff    	sub	sp, sp, #0x20
1000003b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b8: 910043fd    	add	x29, sp, #0x10
1000003bc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c0: 52800220    	mov	w0, #0x11               ; =17
1000003c4: 528000a1    	mov	w1, #0x5                ; =5
1000003c8: 97ffffe6    	bl	0x100000360 <__Z17test_compound_modii>
1000003cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d0: 910083ff    	add	sp, sp, #0x20
1000003d4: d65f03c0    	ret
