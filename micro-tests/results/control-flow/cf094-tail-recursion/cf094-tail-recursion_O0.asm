
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf094-tail-recursion/cf094-tail-recursion_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16factorial_helperii>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000368: 910043fd    	add	x29, sp, #0x10
10000036c: b9000be0    	str	w0, [sp, #0x8]
100000370: b90007e1    	str	w1, [sp, #0x4]
100000374: b9400be8    	ldr	w8, [sp, #0x8]
100000378: 71000508    	subs	w8, w8, #0x1
10000037c: 540000ac    	b.gt	0x100000390 <__Z16factorial_helperii+0x30>
100000380: 14000001    	b	0x100000384 <__Z16factorial_helperii+0x24>
100000384: b94007e8    	ldr	w8, [sp, #0x4]
100000388: b81fc3a8    	stur	w8, [x29, #-0x4]
10000038c: 14000009    	b	0x1000003b0 <__Z16factorial_helperii+0x50>
100000390: b9400be8    	ldr	w8, [sp, #0x8]
100000394: 71000500    	subs	w0, w8, #0x1
100000398: b9400be8    	ldr	w8, [sp, #0x8]
10000039c: b94007e9    	ldr	w9, [sp, #0x4]
1000003a0: 1b097d01    	mul	w1, w8, w9
1000003a4: 97ffffef    	bl	0x100000360 <__Z16factorial_helperii>
1000003a8: b81fc3a0    	stur	w0, [x29, #-0x4]
1000003ac: 14000001    	b	0x1000003b0 <__Z16factorial_helperii+0x50>
1000003b0: b85fc3a0    	ldur	w0, [x29, #-0x4]
1000003b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b8: 910083ff    	add	sp, sp, #0x20
1000003bc: d65f03c0    	ret

00000001000003c0 <__Z9factoriali>:
1000003c0: d10083ff    	sub	sp, sp, #0x20
1000003c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c8: 910043fd    	add	x29, sp, #0x10
1000003cc: b81fc3a0    	stur	w0, [x29, #-0x4]
1000003d0: b85fc3a0    	ldur	w0, [x29, #-0x4]
1000003d4: 52800021    	mov	w1, #0x1                ; =1
1000003d8: 97ffffe2    	bl	0x100000360 <__Z16factorial_helperii>
1000003dc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e0: 910083ff    	add	sp, sp, #0x20
1000003e4: d65f03c0    	ret

00000001000003e8 <_main>:
1000003e8: d10083ff    	sub	sp, sp, #0x20
1000003ec: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003f0: 910043fd    	add	x29, sp, #0x10
1000003f4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003f8: 528000a0    	mov	w0, #0x5                ; =5
1000003fc: 97fffff1    	bl	0x1000003c0 <__Z9factoriali>
100000400: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000404: 910083ff    	add	sp, sp, #0x20
100000408: d65f03c0    	ret
