
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar077-boolean-conversion/ar077-boolean-conversion_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_bool_conversioni>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe8    	ldr	w8, [sp, #0xc]
10000036c: 71000108    	subs	w8, w8, #0x0
100000370: 1a9f07e0    	cset	w0, ne
100000374: 910043ff    	add	sp, sp, #0x10
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: d10083ff    	sub	sp, sp, #0x20
100000380: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000384: 910043fd    	add	x29, sp, #0x10
100000388: 52800000    	mov	w0, #0x0                ; =0
10000038c: b9000be0    	str	w0, [sp, #0x8]
100000390: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000394: 97fffff3    	bl	0x100000360 <__Z20test_bool_conversioni>
100000398: b9400be8    	ldr	w8, [sp, #0x8]
10000039c: 72000009    	ands	w9, w0, #0x1
1000003a0: 1a9f0500    	csinc	w0, w8, wzr, eq
1000003a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a8: 910083ff    	add	sp, sp, #0x20
1000003ac: d65f03c0    	ret
