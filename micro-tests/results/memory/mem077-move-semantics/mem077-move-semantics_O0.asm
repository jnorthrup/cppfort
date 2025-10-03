
/Users/jim/work/cppfort/micro-tests/results/memory/mem077-move-semantics/mem077-move-semantics_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9test_movev>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 52800548    	mov	w8, #0x2a               ; =42
100000368: b9000fe8    	str	w8, [sp, #0xc]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b9000be8    	str	w8, [sp, #0x8]
100000374: b9400be0    	ldr	w0, [sp, #0x8]
100000378: 910043ff    	add	sp, sp, #0x10
10000037c: d65f03c0    	ret

0000000100000380 <_main>:
100000380: d10083ff    	sub	sp, sp, #0x20
100000384: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000388: 910043fd    	add	x29, sp, #0x10
10000038c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000390: 97fffff4    	bl	0x100000360 <__Z9test_movev>
100000394: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000398: 910083ff    	add	sp, sp, #0x20
10000039c: d65f03c0    	ret
