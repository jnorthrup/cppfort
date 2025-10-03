
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf086-short-circuit-side-effects/cf086-short-circuit-side-effects_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_side_effectsi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: 71000108    	subs	w8, w8, #0x0
100000374: 540000ad    	b.le	0x100000388 <__Z17test_side_effectsi+0x28>
100000378: 14000001    	b	0x10000037c <__Z17test_side_effectsi+0x1c>
10000037c: 52800148    	mov	w8, #0xa                ; =10
100000380: b9000be8    	str	w8, [sp, #0x8]
100000384: 14000001    	b	0x100000388 <__Z17test_side_effectsi+0x28>
100000388: b9400be0    	ldr	w0, [sp, #0x8]
10000038c: 910043ff    	add	sp, sp, #0x10
100000390: d65f03c0    	ret

0000000100000394 <_main>:
100000394: d10083ff    	sub	sp, sp, #0x20
100000398: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000039c: 910043fd    	add	x29, sp, #0x10
1000003a0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003a4: 528000a0    	mov	w0, #0x5                ; =5
1000003a8: 97ffffee    	bl	0x100000360 <__Z17test_side_effectsi>
1000003ac: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b0: 910083ff    	add	sp, sp, #0x20
1000003b4: d65f03c0    	ret
