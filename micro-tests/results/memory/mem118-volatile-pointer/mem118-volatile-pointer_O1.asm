
/Users/jim/work/cppfort/micro-tests/results/memory/mem118-volatile-pointer/mem118-volatile-pointer_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z21test_volatile_pointerv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 52800548    	mov	w8, #0x2a               ; =42
100000368: b9000fe8    	str	w8, [sp, #0xc]
10000036c: b9400fe0    	ldr	w0, [sp, #0xc]
100000370: 910043ff    	add	sp, sp, #0x10
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: d10043ff    	sub	sp, sp, #0x10
10000037c: 52800548    	mov	w8, #0x2a               ; =42
100000380: b9000fe8    	str	w8, [sp, #0xc]
100000384: b9400fe0    	ldr	w0, [sp, #0xc]
100000388: 910043ff    	add	sp, sp, #0x10
10000038c: d65f03c0    	ret
