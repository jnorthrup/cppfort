
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf050-switch-fallthrough/cf050-switch-fallthrough_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_switch_fallthroughi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b90007e8    	str	w8, [sp, #0x4]
100000374: 71000508    	subs	w8, w8, #0x1
100000378: 54000140    	b.eq	0x1000003a0 <__Z23test_switch_fallthroughi+0x40>
10000037c: 14000001    	b	0x100000380 <__Z23test_switch_fallthroughi+0x20>
100000380: b94007e8    	ldr	w8, [sp, #0x4]
100000384: 71000908    	subs	w8, w8, #0x2
100000388: 54000140    	b.eq	0x1000003b0 <__Z23test_switch_fallthroughi+0x50>
10000038c: 14000001    	b	0x100000390 <__Z23test_switch_fallthroughi+0x30>
100000390: b94007e8    	ldr	w8, [sp, #0x4]
100000394: 71000d08    	subs	w8, w8, #0x3
100000398: 54000140    	b.eq	0x1000003c0 <__Z23test_switch_fallthroughi+0x60>
10000039c: 1400000d    	b	0x1000003d0 <__Z23test_switch_fallthroughi+0x70>
1000003a0: b9400be8    	ldr	w8, [sp, #0x8]
1000003a4: 11000508    	add	w8, w8, #0x1
1000003a8: b9000be8    	str	w8, [sp, #0x8]
1000003ac: 14000001    	b	0x1000003b0 <__Z23test_switch_fallthroughi+0x50>
1000003b0: b9400be8    	ldr	w8, [sp, #0x8]
1000003b4: 11000908    	add	w8, w8, #0x2
1000003b8: b9000be8    	str	w8, [sp, #0x8]
1000003bc: 14000001    	b	0x1000003c0 <__Z23test_switch_fallthroughi+0x60>
1000003c0: b9400be8    	ldr	w8, [sp, #0x8]
1000003c4: 11000d08    	add	w8, w8, #0x3
1000003c8: b9000be8    	str	w8, [sp, #0x8]
1000003cc: 14000004    	b	0x1000003dc <__Z23test_switch_fallthroughi+0x7c>
1000003d0: 12800008    	mov	w8, #-0x1               ; =-1
1000003d4: b9000be8    	str	w8, [sp, #0x8]
1000003d8: 14000001    	b	0x1000003dc <__Z23test_switch_fallthroughi+0x7c>
1000003dc: b9400be0    	ldr	w0, [sp, #0x8]
1000003e0: 910043ff    	add	sp, sp, #0x10
1000003e4: d65f03c0    	ret

00000001000003e8 <_main>:
1000003e8: d10083ff    	sub	sp, sp, #0x20
1000003ec: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003f0: 910043fd    	add	x29, sp, #0x10
1000003f4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003f8: 52800020    	mov	w0, #0x1                ; =1
1000003fc: 97ffffd9    	bl	0x100000360 <__Z23test_switch_fallthroughi>
100000400: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000404: 910083ff    	add	sp, sp, #0x20
100000408: d65f03c0    	ret
