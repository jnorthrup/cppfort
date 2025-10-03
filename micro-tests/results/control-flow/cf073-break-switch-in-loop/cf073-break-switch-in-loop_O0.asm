
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf073-break-switch-in-loop/cf073-break-switch-in-loop_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z22test_break_switch_loopv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z22test_break_switch_loopv+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71002908    	subs	w8, w8, #0xa
100000378: 5400020a    	b.ge	0x1000003b8 <__Z22test_break_switch_loopv+0x58>
10000037c: 14000001    	b	0x100000380 <__Z22test_break_switch_loopv+0x20>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 71001508    	subs	w8, w8, #0x5
100000388: 54000061    	b.ne	0x100000394 <__Z22test_break_switch_loopv+0x34>
10000038c: 14000001    	b	0x100000390 <__Z22test_break_switch_loopv+0x30>
100000390: 14000005    	b	0x1000003a4 <__Z22test_break_switch_loopv+0x44>
100000394: b9400fe8    	ldr	w8, [sp, #0xc]
100000398: 11000508    	add	w8, w8, #0x1
10000039c: b9000fe8    	str	w8, [sp, #0xc]
1000003a0: 14000001    	b	0x1000003a4 <__Z22test_break_switch_loopv+0x44>
1000003a4: 14000001    	b	0x1000003a8 <__Z22test_break_switch_loopv+0x48>
1000003a8: b9400be8    	ldr	w8, [sp, #0x8]
1000003ac: 11000508    	add	w8, w8, #0x1
1000003b0: b9000be8    	str	w8, [sp, #0x8]
1000003b4: 17ffffef    	b	0x100000370 <__Z22test_break_switch_loopv+0x10>
1000003b8: b9400fe0    	ldr	w0, [sp, #0xc]
1000003bc: 910043ff    	add	sp, sp, #0x10
1000003c0: d65f03c0    	ret

00000001000003c4 <_main>:
1000003c4: d10083ff    	sub	sp, sp, #0x20
1000003c8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003cc: 910043fd    	add	x29, sp, #0x10
1000003d0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d4: 97ffffe3    	bl	0x100000360 <__Z22test_break_switch_loopv>
1000003d8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003dc: 910083ff    	add	sp, sp, #0x20
1000003e0: d65f03c0    	ret
