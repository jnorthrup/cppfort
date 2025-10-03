
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf005-complex-condition/cf005-complex-condition_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z22test_complex_conditionii>:
100000360: 7100001f    	cmp	w0, #0x0
100000364: 7a4ac820    	ccmp	w1, #0xa, #0x0, gt
100000368: 1a9fa7e0    	cset	w0, lt
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800020    	mov	w0, #0x1                ; =1
100000374: d65f03c0    	ret
