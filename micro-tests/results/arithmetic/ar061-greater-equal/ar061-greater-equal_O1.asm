
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar061-greater-equal/ar061-greater-equal_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_greater_equalii>:
100000360: 6b01001f    	cmp	w0, w1
100000364: 1a9fb7e0    	cset	w0, ge
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 52800020    	mov	w0, #0x1                ; =1
100000370: d65f03c0    	ret
