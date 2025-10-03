
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf056-switch-range/cf056-switch-range_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_switch_rangei>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: b90007e8    	str	w8, [sp, #0x4]
100000370: 71000108    	subs	w8, w8, #0x0
100000374: 71002508    	subs	w8, w8, #0x9
100000378: 540000e9    	b.ls	0x100000394 <__Z17test_switch_rangei+0x34>
10000037c: 14000001    	b	0x100000380 <__Z17test_switch_rangei+0x20>
100000380: b94007e8    	ldr	w8, [sp, #0x4]
100000384: 71002908    	subs	w8, w8, #0xa
100000388: 71002508    	subs	w8, w8, #0x9
10000038c: 540000a9    	b.ls	0x1000003a0 <__Z17test_switch_rangei+0x40>
100000390: 14000007    	b	0x1000003ac <__Z17test_switch_rangei+0x4c>
100000394: 52800028    	mov	w8, #0x1                ; =1
100000398: b9000fe8    	str	w8, [sp, #0xc]
10000039c: 14000006    	b	0x1000003b4 <__Z17test_switch_rangei+0x54>
1000003a0: 52800048    	mov	w8, #0x2                ; =2
1000003a4: b9000fe8    	str	w8, [sp, #0xc]
1000003a8: 14000003    	b	0x1000003b4 <__Z17test_switch_rangei+0x54>
1000003ac: b9000fff    	str	wzr, [sp, #0xc]
1000003b0: 14000001    	b	0x1000003b4 <__Z17test_switch_rangei+0x54>
1000003b4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003b8: 910043ff    	add	sp, sp, #0x10
1000003bc: d65f03c0    	ret

00000001000003c0 <_main>:
1000003c0: d10083ff    	sub	sp, sp, #0x20
1000003c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c8: 910043fd    	add	x29, sp, #0x10
1000003cc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d0: 528001e0    	mov	w0, #0xf                ; =15
1000003d4: 97ffffe3    	bl	0x100000360 <__Z17test_switch_rangei>
1000003d8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003dc: 910083ff    	add	sp, sp, #0x20
1000003e0: d65f03c0    	ret
