
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf069-goto-state-machine/cf069-goto-state-machine_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_goto_state_machinei>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: b90007ff    	str	wzr, [sp, #0x4]
100000370: 14000001    	b	0x100000374 <__Z23test_goto_state_machinei+0x14>
100000374: b9400fe8    	ldr	w8, [sp, #0xc]
100000378: 35000068    	cbnz	w8, 0x100000384 <__Z23test_goto_state_machinei+0x24>
10000037c: 14000001    	b	0x100000380 <__Z23test_goto_state_machinei+0x20>
100000380: 14000002    	b	0x100000388 <__Z23test_goto_state_machinei+0x28>
100000384: 1400000b    	b	0x1000003b0 <__Z23test_goto_state_machinei+0x50>
100000388: 52800148    	mov	w8, #0xa                ; =10
10000038c: b90007e8    	str	w8, [sp, #0x4]
100000390: b9400fe8    	ldr	w8, [sp, #0xc]
100000394: 35000068    	cbnz	w8, 0x1000003a0 <__Z23test_goto_state_machinei+0x40>
100000398: 14000001    	b	0x10000039c <__Z23test_goto_state_machinei+0x3c>
10000039c: 14000002    	b	0x1000003a4 <__Z23test_goto_state_machinei+0x44>
1000003a0: 14000004    	b	0x1000003b0 <__Z23test_goto_state_machinei+0x50>
1000003a4: 52800288    	mov	w8, #0x14               ; =20
1000003a8: b90007e8    	str	w8, [sp, #0x4]
1000003ac: 14000001    	b	0x1000003b0 <__Z23test_goto_state_machinei+0x50>
1000003b0: b94007e0    	ldr	w0, [sp, #0x4]
1000003b4: 910043ff    	add	sp, sp, #0x10
1000003b8: d65f03c0    	ret

00000001000003bc <_main>:
1000003bc: d10083ff    	sub	sp, sp, #0x20
1000003c0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c4: 910043fd    	add	x29, sp, #0x10
1000003c8: 52800000    	mov	w0, #0x0                ; =0
1000003cc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d0: 97ffffe4    	bl	0x100000360 <__Z23test_goto_state_machinei>
1000003d4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d8: 910083ff    	add	sp, sp, #0x20
1000003dc: d65f03c0    	ret
