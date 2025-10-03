
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf006-or-condition/cf006-or-condition_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_or_conditionii>:
100000360: 7100281f    	cmp	w0, #0xa
100000364: 7a4ad820    	ccmp	w1, #0xa, #0x0, le
100000368: 1a9fd7e0    	cset	w0, gt
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800020    	mov	w0, #0x1                ; =1
100000374: d65f03c0    	ret
