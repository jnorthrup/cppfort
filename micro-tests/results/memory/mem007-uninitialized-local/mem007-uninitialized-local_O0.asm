
/Users/jim/work/cppfort/micro-tests/results/memory/mem007-uninitialized-local/mem007-uninitialized-local_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_uninitializedv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 52800548    	mov	w8, #0x2a               ; =42
100000368: b9000fe8    	str	w8, [sp, #0xc]
10000036c: b9400fe0    	ldr	w0, [sp, #0xc]
100000370: 910043ff    	add	sp, sp, #0x10
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: d10083ff    	sub	sp, sp, #0x20
10000037c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000380: 910043fd    	add	x29, sp, #0x10
100000384: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000388: 97fffff6    	bl	0x100000360 <__Z18test_uninitializedv>
10000038c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000390: 910083ff    	add	sp, sp, #0x20
100000394: d65f03c0    	ret
