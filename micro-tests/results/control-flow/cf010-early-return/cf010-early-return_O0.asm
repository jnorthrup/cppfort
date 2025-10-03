
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf010-early-return/cf010-early-return_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_early_returni>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: 36f80088    	tbz	w8, #0x1f, 0x10000037c <__Z17test_early_returni+0x1c>
100000370: 14000001    	b	0x100000374 <__Z17test_early_returni+0x14>
100000374: b9000fff    	str	wzr, [sp, #0xc]
100000378: 1400000b    	b	0x1000003a4 <__Z17test_early_returni+0x44>
10000037c: b9400be8    	ldr	w8, [sp, #0x8]
100000380: 71019108    	subs	w8, w8, #0x64
100000384: 540000ad    	b.le	0x100000398 <__Z17test_early_returni+0x38>
100000388: 14000001    	b	0x10000038c <__Z17test_early_returni+0x2c>
10000038c: 52800c88    	mov	w8, #0x64               ; =100
100000390: b9000fe8    	str	w8, [sp, #0xc]
100000394: 14000004    	b	0x1000003a4 <__Z17test_early_returni+0x44>
100000398: b9400be8    	ldr	w8, [sp, #0x8]
10000039c: b9000fe8    	str	w8, [sp, #0xc]
1000003a0: 14000001    	b	0x1000003a4 <__Z17test_early_returni+0x44>
1000003a4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003a8: 910043ff    	add	sp, sp, #0x10
1000003ac: d65f03c0    	ret

00000001000003b0 <_main>:
1000003b0: d10083ff    	sub	sp, sp, #0x20
1000003b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b8: 910043fd    	add	x29, sp, #0x10
1000003bc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c0: 52800640    	mov	w0, #0x32               ; =50
1000003c4: 97ffffe7    	bl	0x100000360 <__Z17test_early_returni>
1000003c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003cc: 910083ff    	add	sp, sp, #0x20
1000003d0: d65f03c0    	ret
