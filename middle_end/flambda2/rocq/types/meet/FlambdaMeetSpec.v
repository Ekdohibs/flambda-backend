(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)



(* What the meet of two Flambda types guarantees: the contract.

   This file is to be reviewed: what is asked of the inputs of a meet
   ([meet_inputs_well_formed]), the contracts [meet_sound],
   [add_equation_sound], [meet_expanded_head_sound],
   [meet_expanded_head_keeps] and [join_scoped_envs_sound], and the
   statements at the end. A contract here is a proposition about a
   function of the type that the compiler's function has. The
   functions are FlambdaMeet.v, with FlambdaMeetEnv.v and
   FlambdaExpandHead.v. That they have their contracts is
   [transcription_sound] of [meet_to_prove], below. It is not among
   the statements of this file, because its proof needs the files
   that come after this one: it is stated and proved in
   FlambdaMeetClosuresSpec.v ([meet_to_prove_sound_statement]), for
   every join that has [join_scoped_envs_sound] and every fuel. The
   statements that are here are proved in
   FlambdaMeetProofs.v. FlambdaMeetExamplesSpec.v has the examples of
   this header, run on the functions: the input on which the
   compiler's meet is not sound, a join that has the contract asked of
   it, an equation that forgets, a meet through a name that holds
   poison.

   The compiler. middle_end/flambda2/types/meet_and_n_way_join.ml (the
   functions from [meet] to [meet_function_type], and the helpers
   before them), types/meet.ml, types/env/meet_env.ml,
   types/expand_head.ml, and flambda2_types.ml for
   [Typing_env.add_equation]. The compiler has a second file with a
   meet, types/meet_and_join.ml, the one it runs by default. The two
   have the same functions but for the join that four of them call,
   and two helpers that differ, [meet_disjunction] and [meet_row_like],
   which join the environments of the alternatives of a disjunction.
   So there is one contract, here; the traversal is transcribed once,
   with that join as a parameter ([join_scoped_envs]); and it is
   instantiated first with the n-way join, which is another
   workstream's (T5). The other file is not the reference.

   The contract, in short ([meet_sound]). [meet tenv t1 t2] answers
   bottom, or a type [t] and an environment [tenv'], which is [tenv]
   with equations added and variables defined. If a valuation
   satisfies [tenv] and gives a value [v] both the types [t1] and
   [t2], then:
   - the answer is not bottom. So bottom is answered only if no value
     has both types under any valuation that satisfies the environment
     ([meet_bottom_sound]);
   - some valuation satisfies [tenv'], gives [v] the type [t], and
     agrees with the first on every variable that [tenv] knows: the
     same value, or none in both ([extended_from]). The variables that
     [tenv'] knows and [tenv] does not are those that the meet defined;
     the valuation may give each a value, and one valuation accounts
     for all the equations that the meet added. So what the first
     valuation satisfied, of environments and types over the variables
     that [tenv] knows, the second satisfies: satisfaction depends on
     the values of the variables mentioned only
     (FlambdaTypingEnvSpec.v). The meet of an alternative of a
     disjunction relies on it: the valuation that it answers has to
     satisfy the environment of the whole meet still, for the join.
   What is known is kept and what is new is true. Nothing is said of
   how much the meet knows: see "Precision" below.

   What satisfies an environment. The contracts are about
   [typing_env_satisfied_strictly] of FlambdaTypingEnvSpec.v: every
   name that has a value in the valuation has the type that the
   environment gives it, and aliases have the same value. They are not
   about [typing_env_satisfied], which asks nothing of the type of a
   name that is exempt because of poison (decision D26). Three things
   about the strict notion matter here.
   - Poison is a value like another. A meet reads the types of names:
     of the name that a type is an alias to, of the name of which a
     number is the result of a relation, of the names of an environment
     extension. If such a name holds poison and its type is not true of
     poison, the meet deduces from a type that does not hold. Example:
     [x] holds poison and the environment has [x : null]. The value
     poison has the types [= x] and [any_non_null_value], and their
     meet is bottom. With [x] and [y] holding poison,
     [x : Variant (is_int = z)], [y : Variant (is_int = z')], [z]
     holding 1 and [z'] holding 0, the meet of [= x] and [= y], both
     of which poison has, makes [z] and [z'] aliases. And the name
     need not be the one that an equation is added to: it may be
     reached through a field. So the premise "the value of the name is
     not poison" on the name of [add_equation] is not enough, and the
     contracts ask that every name has its type. A name may hold
     poison if its type is true of poison (the unknown type is). There
     is a CR at [add_equation_sound].
   - The relations [Is_int] and [Get_tag] are not true of a scrutinee
     that holds poison (decision D26, as revised). From "[t] is
     [Get_tag] of [p]" and the numbers that [t] may be, meet gives [p]
     a type of blocks, and from [Is_int] a type of blocks or of
     immediates ([reduce_inverse_relations]); and it reads a relation
     on a name that is an alias of a constant as a number, 1 for
     [Is_int] of a poison constant ([Relation.of_const]). None of
     these types is true of poison. So these two relations must not
     hold of poison, or the strictly satisfied environment
     [p : unknown], [t : Get_tag of p], with [p] holding poison and
     [t] holding 0, would make the meet of [= t] with the number 0
     unsound. FlambdaTypeMeaningSpec.v says so ([relation_holds] has
     no clause for poison in these two cases).
     [Is_null] relates poison to 0, as it does every value that is not
     null, and stays so: from 0 meet deduces the type of the values
     that are not null, which poison has.
     The relations that unboxing gives the [Is_int] and tag parameters
     of a field of an inactive case, which are on a parameter that
     holds poison, are part of what [typing_env_satisfied] exempts,
     and the contracts say nothing of them. (Discrepancy T4-5.)
   - A name may have no value. The n-way join defines, in the
     environment that it answers, variables that come from one of the
     environments joined, with the types they have there. Under a
     valuation of another of the environments joined no value need
     have such a type. Such a variable has no value then, and nothing
     is asked of it, whatever its type: it may be "[Get_tag] of [p]"
     with [p] holding poison. A meet does not reach the type of a name
     with no value: it makes a deduction only on behalf of a value
     that it has in hand, and what tells it that a name has a value is
     the type that the value in hand has. An alias to a name gives the
     name that value; a relation of a naked immediate is "some value
     of the name is so related" (FlambdaTypeMeaningSpec.v); an
     equation of an extension is "the name has a value of this type".
     The extensions of an alternative are added to its environment
     before it is known that the value is of that alternative; where
     it is not, the environment of the alternative is not claimed to
     be satisfied, and only the other is given to the join. The
     functions that add to an environment with no value in hand take
     the premise that the name has a value of the type
     ([add_equation_sound], [add_env_extension_sound],
     [add_equations_on_params_sound]).
   - No equation is added on a name that has no value
     ([keeps_equations_valued]). Meet makes extensions of the levels
     of the environments of its alternatives ([extension_of_level],
     FlambdaMeet.v) and puts them in the type that it answers, for the
     cases of a row-like structure and the two sides of a variant. An
     extension is true where each of its names has a value of the type
     of its equation, and it has to mean that: a later meet adds the
     extension of a case to an environment and drops the case if that
     is bottom, which is right only if the names have values of the
     types. So an equation of a level that is cut has to be on a name
     that has a value. Each contract here says that it is kept: if the
     equations of the levels after a scope were on names that had a
     value, they are still, in the environment and under the valuation
     answered. Meet adds an equation only to a name that it reached
     from the value in hand. The join is asked for the same, and the
     compiler's n-way join has it not: see below.
   - The static environment is valid ([statics_valid],
     FlambdaClosuresSpec.v): what each symbol is bound to is valid,
     and so a static set of closures names a symbol for each of its
     function slots. The states of a run of a well-formed unit have
     it. Without it the meet of two closures types is wrong: a type
     says of the closure at a function slot [g] of the set what holds
     of the projection to [g] if there is one (the meaning of a
     product: FlambdaTypeMeaningSpec.v), and from a symbol of a
     static set that names no symbol for [g] there is none, though
     the set has [g]. Such a value has two types whose closure types
     at [g] have no value in common, and the meet of the two is
     bottom. There is a CR at [meet_sound].
   An environment that is satisfied with the names that poison exempts
   set aside, by a valuation in which no name holds poison, is
   satisfied strictly ([satisfied_no_poison_strictly_statement],
   FlambdaTypingEnvSpec.v). The environment of a handler whose
   parameters unboxing gives poison to on some paths is not, on those
   paths: the contracts say nothing there.

   What is asked of the inputs ([meet_inputs_well_formed]).
   - The two types have one kind. A meet of types of two kinds is a
     fatal error of the compiler ([Meet_env.meet]: "Kind mismatch upon
     meet"), never bottom; a fatal error is [None] here
     ([meet_kinds]). An environment that is bottom is tested for first,
     and then the answer is bottom whatever the kinds
     ([meet_bottom_env]).
   - The types, and the types of the environment, are well formed:
     [type_well_formed] (FlambdaTypeWellFormedSpec.v). What meet
     needs of it: (a) the product of a case of blocks is no wider
     than the size of its index; (b) the function types, the closure
     types and the value slot types of a case of closures are of slots
     among those of its index, and an exact index has the slot of the
     case among its closures; (c) a shape is written in one way; (d)
     the kinds: the immediates of a variant are of kind naked
     immediate, the contents of a boxed number of the kind boxed, the
     closure types of kind value; (e) a set of numbers that is a head
     is not empty ([set_meet] answers one of its inputs where one set
     is included in the other, so an empty set would come back as an
     answer that is not bottom). The compiler's creation functions
     give (a) to (e) under preconditions that not all of them check,
     and its raw constructors check nothing (discrepancy T1-5): here
     the model is stricter than the compiler (decision D32), and the
     file that defines the predicate says so at each condition. The
     n-way join did not keep (b), until it was changed to keep a
     function type only if every input has one: it kept a function
     type under an index that does not say that the set has the
     function (T1-4). On such a type the meet answers bottom for two
     function types of one slot with different code, and a closure of
     a set without that slot has both types (discrepancy T4-3).
     That a type is not an alias to a poison constant is not asked:
     the compiler makes such types
     ([Typing_env.type_simple_in_term_exn]), and nothing goes wrong
     with them, the only value of such a type being poison.
   - The names that the types mention are known to the environment,
     at the kinds at which the types use them ([type_fits]): its
     variables are bound there, each alias is at the kind of the type
     that the environment has for its target, and the variables that
     hold the results of [Is_int], [Get_tag] and [Is_null] are naked
     immediates there ([type_kinded]); and each name that has an
     equation in an extension inside the type is of the kind of the
     type of its equation ([type_equations_kinded]). Meet itself does
     not need the kinds of the aliases: where they are wrong it stops
     with a fatal error ([Typing_env.find] with a kind,
     [Meet_env.add_equation]). The join that it calls does (below), so
     meet has to be given them and to give them back. The kinds of the
     names of equations it needs itself, to keep the environment
     kinded: where it adds the extension of a case, it checks the kind
     of a name only where the type is not an alias and the canonical
     element of the name is a name (FlambdaMeetEnvSpec.v). There is a
     CR at [type_fits]. With it go two conditions on the environment,
     which with the kinds of the aliases of its types make it kinded
     ([typing_env_kinded], FlambdaTypingEnvKindedSpec.v, with its own
     CRs): two simples that it knows to be equal are of one kind
     ([aliases_kinded]), and a symbol that has a type has one of the
     kind of values ([symbols_of_kind_value]). A meet that makes a
     name an alias of another writes on it an alias type of the kind
     of the name, and that type is kinded only if the other is of
     that kind; and [Meet_env.add_alias] takes a symbol to be of the
     kind of values. The kinds of the names of the equations inside
     the types of the environment are a condition beside it
     ([types_equations_kinded], in the same file).
   - The environment is well formed: it has the invariant of typing
     environments ([typing_env_invariant],
     FlambdaTypingEnvInvariantSpec.v): binding times, levels and
     scopes, the aliases, and that its types mention only variables
     that it knows; and its types are well formed
     ([types_well_formed], in the same file, beside the invariant).
     The invariant is stricter than what the compiler checks, and
     that file says where. It holds of the environments
     that a meet goes through, also between the two steps in which a
     name is made an alias of another and is given the alias for a
     type: the statements of that file about [Typing_env.add_alias]
     and [Typing_env.replace_equation] are for those steps.
   - No name has a type that is obviously bottom, unless the
     environment is bottom ([no_bottom_types]). A name with such a
     type has no value. The n-way join passes over an obviously bottom
     type where it joins the types that a name has in the environments
     joined, and can then make the name an alias of what it is in the
     others: its witness would have to give a value to a name that had
     none, and the agreement above would be lost. Meet keeps the
     condition: it answers bottom for the whole environment where an
     equation that it adds is obviously bottom. The compiler's join
     needs the condition, and the condition is not enough for it: a
     type that is not obviously bottom may have no value either, and
     that join passes over it in the same way (below, "The compiler's
     n-way join"). There is a CR at [no_bottom_types].
   The answer is well formed in the same sense and of the kind of the
   inputs, and the environment answered knows what the first knew
   ([meet_keeps_well_formed]), so that a meet can follow a meet.

   The join that meet is given ([join_scoped_envs_sound]). It is
   [Join_env.cut_and_n_way_join] with its [n_way_join_type], called
   by [meet_disjunction] (the two sides of a variant, and a value that
   may be null) and by [meet_row_like] (the cases of a row-like
   structure): each alternative is met in an environment of its own,
   one scope further than the environment of the meet, and the join
   makes of these one environment. It is given the meet of expanded
   types as an argument, and its contract supposes that this argument
   has its own, which is in two parts: what it says of values
   ([meet_expanded_head_sound]) and what it keeps of the environment
   whatever the values ([meet_expanded_head_keeps]). What is asked, as
   agreed with T5 (its contract is FlambdaJoinSpec.v):
   - a valuation that satisfies the environment of the meet and one of
     the environments joined extends to one that satisfies the result,
     and that agrees with it on every variable that the environment
     joined knows; one valuation for all that the join defines. The
     variables that the result takes from the other environments
     joined may have no value in it; the result is then to have no
     equation on them ([keeps_equations_valued]; the join is given
     that the equations that the environment joined has after the
     scope of the meet are on names that have a value).
     The premise is about one valuation and both environments. It is
     not that whatever satisfies an environment joined satisfies the
     environment of the meet: that is false of the environments that
     meet makes. Adding an equation can forget: with
     [x : Value (unknown, null or not, is_null = z)], adding
     [x : any_non_null_value] answers [x : any_non_null_value] and
     says nothing of [z], which was 0 if [x] was not null
     ([meet_disjunction], the case of one alternative that is bottom).
     And the join needs the environment of the meet to be satisfied:
     for a name that every environment joined has an equation on, it
     answers the meet of the type joined with the type that the
     environment of the meet records. The valuation in hand satisfied
     that environment before the alternative was met, and the
     valuation that the alternative answers agrees with it on what
     that environment knows, so it satisfies it still: that is what
     meet gives, and why the agreement has to be exact;
   - the result is the first environment with equations and
     definitions added: its scope, its earlier levels and what it knew
     are unchanged ([meet_env_extends]);
   - a variable that the result knows and that none of the environments
     given knows is new: it is not among the names to avoid
     ([Meet_env.names_to_avoid]) of the environment given, and is
     among those of the result. And it may not occur in terms: the
     result has it at a name mode that is not the normal one. Nothing
     else is said of the variables made: not which, nor in what order.
     Meet itself makes no variable; it defines again, in the result,
     the variables that the alternatives defined (see "Known
     differences"), in the name mode of the variables that exist only
     in types;
   - the result is well formed, and kinded;
   - a [Simple] that the result knows is, in the result, of the kind
     that an environment joined has it at. After the join the meet
     defines, in the result, the variables of the alternatives that
     the result does not know, and answers types that fit the
     environments of the alternatives: they are to fit the result.
   Meet gives the join environments that it made from the first by
   adding definitions and equations ([scoped_env_of]): each fits, has
   every name of the first with its binding time and a type of the
   same kind, and knows no symbol that the first does not (a meet
   defines no symbol; the join may then put an alias to a symbol of
   an environment joined into a type of the result); and no two of
   them have one [Simple] at two kinds. The
   variables that two alternatives define are different variables:
   the second alternative is given the names to avoid of the first,
   and a meet has its new variables among the names to avoid that it
   answers and not among those that it was given
   ([new_names_avoided], a clause of [meet_expanded_head_keeps]). The
   same clause says that the new variables of a meet may not occur in
   terms; so a meet adds no variable that may
   ([meet_to_prove_keeps_terms_statement], FlambdaMeetTermsSpec.v).

   The compiler's n-way join. It has not the first clause as it is
   stated (discrepancy T5-10; FlambdaJoinSpec.v states what it has).
   It takes a variable that an environment has a type for to have a
   value there, and a valuation that satisfies an environment strictly
   may give it none. Two things follow.
   - Where the type joined for a name [x] is a variant with a variable
     [r] for [Is_int], the join gives [r] the type "[Is_int] of [x]"
     ([recover_inverse_relations]), also where [x] has no value in the
     run and [r] has one, and that type is true of no number then.
   - The join of the types that a name has in the environments joined
     passes over a type that it finds to have no value: one that is
     obviously bottom, or a head of naked immediates that its
     relations reduce to nothing ([expand_head], by
     [reduce_head_of_kind_naked_immediate]). The name then gets what
     it is in the other environments joined, an alias of a constant
     say, also on a run through the environment in which it has no
     value. [no_bottom_types] rules out the first kind of type only.
     A name whose type is "0 or 2, and the result of [Is_int] on [p]",
     in an environment where [p] is a tagged immediate, has a type
     that is not obviously bottom, and no value; given it in one
     environment and the constant 1 in the other, both joins of the
     compiler answer that the name is the constant 1 (the stream of
     differential tests ran it).
   A variable that the join makes or brings in, with its type, may
   have no value on a run through one of the environments joined; the
   result then has an equation on a name with no value, and the
   extension that an enclosing meet cuts from it is false of a value
   that is in its case. So the compiler's join has not what is asked
   here of equations either ([keeps_equations_valued], in the first
   clause), unless every variable that has an equation in the level
   that the join adds has a value on the run: a fourth condition.
   So the compiler's join has the first clause only of the runs that
   have the values that it reads: three conditions on a run, stated
   in FlambdaJoinSpec.v ([scrutinees_have_values],
   [joined_variables_have_values], [imported_variables_have_values]),
   and the fourth.
   The clause here is not weakened for them: a meet that follows
   reads a relation on behalf of the value of [r] and adds an
   equation to [x], whose type nothing ties to anything if it has no
   value, so with [r] exempt that meet could answer bottom for a
   value that has both types. The meet is to be proved with the
   clause as a hypothesis on the join that it is given. Its instance
   for the compiler's own join is owed, and it has an obligation
   that nothing here discharges: that every join that a meet makes
   is made on runs that have these conditions. They are to be
   carried through the meet as conditions on the run, and are
   hypotheses of that instance wherever the meet is not shown to
   establish them, with a CR there: nothing in the compiler is known
   to establish the first and the third, and the first fails of a
   program that the compiler, with the n-way join, then compiles to
   code that answers a wrong number (the reproduction of discrepancy
   T5-10).

   Code ids. A function type names its code exactly
   (FlambdaTypeMeaningSpec.v); the code age relation is not in the
   model's environment. So the model's [meet_code_id] answers bottom
   for two different code ids, which is right for that meaning where
   the set has the function (condition (b) above). The compiler
   answers the newer of the two if one is a later version of the other
   ([Code_age_relation.meet]), and either input if it cannot tell. The
   contract claims nothing of that: the model and the compiler part
   ways on two code ids that the relation relates, and the differential
   tests are to leave those inputs out or expect the difference
   (discrepancy T4-4). A claim that the code named by a meet is the
   code of the closure at run time needs code age (workstream S4).

   Allocation modes and uniqueness mean nothing in the meaning of
   types, so the contract says nothing of [meet_alloc_mode] or of
   [is_unique]. They are transcribed as they are.

   Physical equality. The compiler's meet answers [Left_input],
   [Right_input] or [Both_inputs] so as to give back an input unchanged
   and keep sharing; [meet_inverse_relations] tests [==] as a shortcut.
   Neither changes which type is answered. The model has no physical
   equality (decision D30): the transcription follows the slow path,
   and the contract is about the type answered, by [extract_value].

   Precision. The contract says that a meet loses no value. It does
   not say that the type answered has no more values than each input
   has, and that is not true of the compiler's meet: an equation on a
   name is dropped while another equation on that name is being added
   ([adding_equations_for_names]); a case that is impossible falls back
   to the case of the other tags ([meet_knowns], with a CR in the
   compiler); the extensions of the cases are joined; an alternative
   that is bottom takes with it what its variable said (the example
   above). What precision is promised, and for which inputs, is to be
   stated apart.

   Termination. The functions of meet call each other through the
   environment: a meet with an alias reads the type that the
   environment has for the name and meets that. It is not a recursion
   on the two types. In the compiler it stops because an equation on a
   name is dropped while one is being added on that name
   ([Meet_env.add_or_delay_concrete_equation_on_canonical_name]) and,
   between two such additions, the types get smaller; nothing states
   it. The transcription is one fixpoint on a number, the fuel, with
   [None] where it runs out, and gives itself, with the fuel that is
   left, to the join. The contracts are about answers that are not
   [None], so they need no fuel; that some fuel is enough for every
   well formed input is a statement of its own, to come, and is not
   needed for soundness.

   The step that is not sound. [reduce_inverse_relations], in its case
   [Get_tag]: when the numbers of a naked immediate that is [Get_tag]
   of a name become known, it adds to the name the equation
   [blocks_with_these_tags tags], a type with no immediate, [tags]
   being those of the numbers that are tags of blocks
   ([Tag.create_from_targetint]: 0 to 255). The relation holds of a
   name that is an immediate, with any number: unboxing records it of
   a parameter that may be one, and passes the tag 0 where it is
   (FlambdaTypeMeaningSpec.v). So the equation added is not true
   ([get_tag_reduction_unsound_statement]), and no meet that makes
   this step has [meet_sound]. The input on which it fails: the
   environment [p : Variant (get_tag = tag)],
   [tag : (numbers unknown, Get_tag p)], as
   [Typing_env.add_get_tag_relation] makes it; the valuation in which
   [p] is the tagged immediate 7 and [tag] is 0, which satisfies it
   strictly; the meet of [= tag] and of the type of the naked
   immediate 0, both of which the value 0 has. The environment
   answered has [p : blocks_with_these_tags [0]], which no valuation
   that keeps the value of [p] satisfies. What would make it sound: to
   add the type of the immediates and of the blocks with one of the
   tags ([immediate_or_blocks_with_these_tags], FlambdaMeet.v), which
   is what the relation gives ([get_tag_reduction_repaired_statement]);
   in the compiler, [~immediates:Unknown] in place of the bottom type
   of naked immediates in the type that this case builds. Nothing is
   lost where the name is already known to be a block. The contract is
   not weakened for this: the transcription takes the type added in
   this case as a parameter, so that the function as the compiler has
   it can be run and compared ([compiler_meet]), and the function with
   the type that is true can be proved ([meet_to_prove]).
   (Discrepancies T1-1, R3-2; in both files.)
   Two things that statement needs of the value, which the relation
   gives. It is not poison (above). And if it is a block, its tag is
   one of the numbers that are tags: the step drops the others ("no
   blocks exist with this tag"), and a block of the model may have
   any number for its tag. That the blocks of a run have tags in
   range is a rule of well formed terms (decision D23), which the
   contract of a meet, being about any value, has not; so the
   relation [Get_tag] relates a block to its tag only if the tag is
   in range, and the rule is used where a relation is made from a
   primitive. (Discrepancy T4-6.)

   Known differences between what the compiler does and what is said
   here, each recorded: T4-1 (after the join, meet defines again the
   variables that the join imported: a meet that does so does not keep
   an environment well formed, and [meet_keeps_well_formed] is to be
   proved of one that defines only what the result does not know:
   [define_again], FlambdaMeet.v), T4-2 (a meet through a name that
   holds poison), T4-3 (function types of slots that the index does
   not have), T4-4 (code age), T4-5 (a relation on a scrutinee that
   holds poison), T4-6 (tags out of range), T4-9 (where
   [meet_row_like] keeps one case only it does not define, in the
   environment that it answers, the variables that the meet of that
   case defined, and the join that it calls on that one environment
   need not keep them: the type answered can mention a variable that
   the environment answered does not know, and
   [meet_keeps_well_formed] is to be proved of the meet that lifts
   them: [lift_variables_of_single_case], FlambdaMeet.v). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet.
Import ListNotations.

(* ======================================================================== *)
(* What a meet may change of a valuation                                    *)
(* ======================================================================== *)

(* What a meet or a join may change of a valuation: nothing, on the
   variables that the environment [tenv] knows. A variable that had no
   value has none. The variables that are new may have any value, or
   none. *)
Definition extended_from (tenv : Typing_env.t) (valuation valuation' : env)
    : Prop :=
  ∀ x : variable,
    Typing_env.mem None tenv (Name_var x) = true ->
    lookup x valuation' = lookup x valuation.

(* Every equation that the levels of an environment after the scope
   [cut_after] have is on a name that has a value. Those are the
   equations of the level that [Typing_env.cut] answers for that
   scope, and so of the extension that is made of it. *)
Definition equations_valued (tenv : Typing_env.t) (cut_after : Scope.t)
    (valuation : env) : Prop :=
  ∀ (one_level : Typing_env.One_level.t) (n : name) (ty : flambda_type),
    one_level ∈ Typing_env.current_level tenv :: Typing_env.prev_levels tenv ->
    (cut_after < Typing_env.One_level.scope one_level)%Z ->
    (n, ty)
    ∈ Typing_env_level.equations (Typing_env.One_level.level one_level) ->
    ∃ v : value, value_of_name valuation n = Some v.

(* An equation is added only on a name that has a value: whatever the
   scope, if the equations after it were on names that had a value,
   they are still. *)
(* CR mshinwell: this is what the compiler relies on wherever it makes
   an extension of the level of an environment and later adds that
   extension to another environment (the cases of a row-like structure,
   the two sides of a variant, the join of the environments of a
   continuation): the equation of a name that has no value in a run is
   then added in that run, and its meet with what is known of the name
   may be bottom. Meet keeps it, as far as its own equations go. The
   compiler's n-way join does not: it gives an equation to a variable
   that it makes for the simples that a type has in the environments
   joined, which has no value on a run through an environment where
   there is no such simple or the simple has none; and to a variable
   that it brings in from some of the environments joined, which has
   none on the runs through the others. That every variable with an
   equation in the level that the join adds has a value at the
   environment of the run is a fourth condition on the runs of the
   compiler's join, beside the three of FlambdaJoinSpec.v
   (discrepancies T5-5 and T5-10). *)
Definition keeps_equations_valued (tenv tenv' : Typing_env.t)
    (valuation valuation' : env) : Prop :=
  ∀ cut_after : Scope.t,
    equations_valued tenv cut_after valuation ->
    equations_valued tenv' cut_after valuation'.

(* ======================================================================== *)
(* The functions                                                            *)
(* ======================================================================== *)

(* The type of [Meet.meet] (meet.ml, and [Meet_and_n_way_join.meet] as
   it is exposed): bottom, or a type and the environment extended.
   [None] is a fatal error of the compiler, or a meet that was not
   given enough fuel, or the one state at which the model stops and
   the compiler does not (FlambdaMeetEnv.v, "A name that is no longer
   canonical"; discrepancy T4-10). *)
Definition meet_function : Type :=
  Typing_env.t -> flambda_type -> flambda_type ->
  option (Or_bottom.t (flambda_type * Typing_env.t)).

(* The type of [Typing_env.add_equation] of flambda2_types.ml: the
   environment in which the name has the type given as well as the type
   it had. Where the two have no value in common the environment
   answered is bottom ([Typing_env.make_bottom]). *)
Definition add_equation_function : Type :=
  Typing_env.t -> name -> flambda_type -> option Typing_env.t.

(* The types of [Typing_env.add_env_extension] and of
   [Typing_env.add_equations_on_params] of flambda2_types.ml. *)
Definition add_env_extension_function : Type :=
  Typing_env.t -> env_extension -> option Typing_env.t.

Definition add_equations_on_params_function : Type :=
  Typing_env.t -> list (variable * flambda_kind) -> list flambda_type ->
  option Typing_env.t.

(* The type of the join that meet is given:
   [Join_env.cut_and_n_way_join ~n_way_join_type ~meet_expanded_head
   ~cut_after initial_env initial_tenv scoped_envs]. *)
Definition join_scoped_envs : Type :=
  meet_expanded_head -> Scope.t -> Meet_env.t -> Typing_env.t ->
  list Typing_env.t -> option Meet_env.t.

(* ======================================================================== *)
(* Kinds                                                                    *)
(* ======================================================================== *)

(* That a [Simple] is of a kind in an environment ([simple_of_kind]),
   that a type is kinded there ([type_kinded]: each of its aliases is
   at the kind that the environment has for its target, and its
   relation variables are naked immediates there), and that an
   environment is kinded ([typing_env_kinded]: its types are, two
   [Simple]s that it knows to be equal are of one kind, and a symbol
   that has a type has one of the kind of values) are the typing
   environment's (FlambdaTypingEnvKindedSpec.v). So are the kinds of
   the names of the equations inside a type ([type_equations_kinded])
   and inside the types of an environment
   ([types_equations_kinded]). *)

(* No name has a type that is obviously bottom, in an environment that
   is not bottom. *)
(* The model is stricter than the compiler here (decision D32). The
   compiler does make environments that are not bottom in which a name
   has the bottom type: [Meet_env.add_env_extension_maybe_bottom],
   for the binary join (types/join_levels_old.ml). The meet and the
   n-way join do not call it: where an equation has no value they make
   the whole environment bottom ([Meet_env.add_equation],
   [Meet_env.add_env_extension]) or answer bottom
   ([Meet_env.add_equation_strict]). *)
(* CR mshinwell: this is asked of every environment that a meet is
   made in, because the n-way join that the meet calls is not sound
   without it: the join of types passes over an obviously bottom type,
   and the join of environments can then give the name what it is in
   the other environments joined, where it has no value in this one.
   It is not enough for the compiler's join to be sound, which passes
   in the same way over a type that is not obviously bottom and has
   no value (the header, "The compiler's n-way join"). It is not known
   that every environment that Simplify builds with the n-way join has
   it. Discrepancy T5-10. *)
Definition no_bottom_types (tenv : Typing_env.t) : Prop :=
  Typing_env.is_bottom tenv = false ->
  ∀ (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    is_obviously_bottom ty = false.

(* ======================================================================== *)
(* What is asked of the inputs                                              *)
(* ======================================================================== *)

(* A type that may be met in an environment: it is well formed
   ([type_well_formed], FlambdaTypeWellFormedSpec.v: the conditions on
   a type that the compiler's functions establish and meet relies
   on), the environment knows its variables, and it is kinded there:
   its aliases ([type_kinded]) and the names of the equations of the
   extensions inside it ([type_equations_kinded]) are of the kinds at
   which the type has them (FlambdaTypingEnvKindedSpec.v). *)
(* CR mshinwell: that the type is kinded is a premise of the
   contracts, and it is about kinds, on which the model may not ask
   more than the compiler (decisions D10, D12 and D32). It is not
   claimed that the environments of the compiler have it. A type does
   not say at which kind the target of an alias is bound, and
   ill-kinded code, which Simplify must accept, may give a name an
   alias at another kind. The compiler does not check it where a type
   is made. It stops with a fatal error where it follows such an alias
   ([Typing_env.find] with a kind; "Kind mismatch upon meet"), so a
   meet answers nothing on one, and soundness does not need the
   premise. The n-way join needs it of the environments that it is
   given (it makes a variable of the kind of the types joined for the
   [Simple]s that they are aliases to, and does not look at their
   kinds), so the meet that calls it needs it of its own inputs.
   Whether an environment that is not kinded can reach a meet is not
   known. Discrepancy T5-9. The kinds of the names of equations the
   meet needs itself: [Meet_env.add_equation] does not check the kind
   of a name where the type is an alias or the canonical element of
   the name is a constant (FlambdaMeetEnvSpec.v). *)
Definition type_fits (tenv : Typing_env.t) (t : flambda_type) : Prop :=
  type_well_formed t = true
  ∧ (∀ x : variable,
      x ∈ free_variables t -> Typing_env.mem None tenv (Name_var x) = true)
  ∧ type_kinded tenv t
  ∧ type_equations_kinded tenv t.

(* An environment in which a meet may be made: it has the invariant of
   typing environments, its types are well formed, it is kinded, the
   names of the equations inside its types are of the kinds of their
   types ([types_equations_kinded], FlambdaTypingEnvKindedSpec.v),
   and no name has an obviously bottom type in it unless it is
   bottom. Each of its types then fits it ([fits_types_statement]). *)
Definition typing_env_fits (tenv : Typing_env.t) : Prop :=
  typing_env_invariant tenv
  ∧ types_well_formed tenv
  ∧ typing_env_kinded tenv
  ∧ types_equations_kinded tenv
  ∧ no_bottom_types tenv.

Definition meet_inputs_well_formed (tenv : Typing_env.t)
    (t1 t2 : flambda_type) : Prop :=
  typing_env_fits tenv
  ∧ type_fits tenv t1
  ∧ type_fits tenv t2
  ∧ kind t1 = kind t2.

(* An environment extension that may be added to an environment: the
   environment knows the names it has equations on, and their types
   fit. *)
Definition env_extension_fits (tenv : Typing_env.t)
    (extension : env_extension) : Prop :=
  match extension with
  | Env_extension equations =>
    ∀ (n : name) (ty : flambda_type),
      Keyed.lookup name_eqb n equations = Some ty ->
      Typing_env.mem None tenv n = true ∧ type_fits tenv ty
  end.

(* ======================================================================== *)
(* The contract of meet                                                     *)
(* ======================================================================== *)

(* A meet loses no value and what it adds is true. If a valuation
   satisfies the environment and gives [v] both types, the meet does
   not answer bottom, and a valuation that agrees with the first on
   what the environment knew satisfies the environment answered and
   gives [v] the type answered. The static environment is valid. *)
(* CR mshinwell: that the static environment is valid is a premise of
   every statement here that says that no value is lost, and of what
   is asked of the join. It was added after review: without it the
   statements are false of the model, though of no run of a
   well-formed unit (a term of the model can allocate a static set
   that is not valid; that a run keeps the static environment valid
   is proved of well-formed units). The reviewer's case is a symbol
   bound to a closure of a static set that has the function slots
   [f] and [g] and names a symbol for [f] only: the projection of
   [g] from it has no value, a closures type asks nothing of the
   closure at [g] then, and two such types whose closure types at
   [g] do not meet are met to bottom ([meet_closures_entry]). The
   other repair is in the meaning of a closures type: that a closure
   type at [g] asks that the projection have a value. The meaning was
   left as it is: a product says something of the components that
   exist, and what exists is for the index to say in a static
   environment that is valid; the index alone does not give the
   symbol. *)
Definition meet_sound (meet : meet_function) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t1 t2 : flambda_type) (v : value)
    (result : Or_bottom.t (flambda_type * Typing_env.t)),
    meet_inputs_well_formed tenv t1 t2 ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation tenv ->
    type_satisfied statics valuation v t1 ->
    type_satisfied statics valuation v t2 ->
    meet tenv t1 t2 = Some result ->
    ∃ (t : flambda_type) (tenv' : Typing_env.t) (valuation' : env),
      result = Or_bottom.Ok (t, tenv')
      ∧ extended_from tenv valuation valuation'
      ∧ typing_env_satisfied_strictly statics valuation' tenv'
      ∧ type_satisfied statics valuation' v t
      ∧ keeps_equations_valued tenv tenv' valuation valuation'.

(* The same, read from the answer: where a meet answers bottom, no
   value has both types under a valuation that satisfies the
   environment. It follows from [meet_sound]
   ([meet_answers_bottom_statement]). *)
Definition meet_bottom_sound (meet : meet_function) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t1 t2 : flambda_type) (v : value),
    meet_inputs_well_formed tenv t1 t2 ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation tenv ->
    meet tenv t1 t2 = Some Or_bottom.Bottom ->
    ~ (type_satisfied statics valuation v t1
       ∧ type_satisfied statics valuation v t2).

(* What is answered may be met again: the type is of the kind of the
   inputs, it and the environment are well formed, and the environment
   knows every name that the first knew. *)
Definition meet_keeps_well_formed (meet : meet_function) : Prop :=
  ∀ (tenv tenv' : Typing_env.t) (t1 t2 t : flambda_type),
    meet_inputs_well_formed tenv t1 t2 ->
    meet tenv t1 t2 = Some (Or_bottom.Ok (t, tenv')) ->
    typing_env_fits tenv'
    ∧ type_fits tenv' t
    ∧ kind t = kind t1
    ∧ ∀ n : name,
        Typing_env.mem None tenv n = true ->
        Typing_env.mem None tenv' n = true.

(* Kinds. In an environment that is not bottom, a meet of types of two
   kinds is a fatal error: it is never bottom, nor a type. *)
Definition meet_kinds (meet : meet_function) : Prop :=
  ∀ (tenv : Typing_env.t) (t1 t2 : flambda_type),
    Typing_env.is_bottom tenv = false ->
    kind t1 <> kind t2 ->
    meet tenv t1 t2 = None.

(* In an environment that is bottom every meet is bottom, and the kinds
   are not looked at. *)
Definition meet_bottom_env (meet : meet_function) : Prop :=
  ∀ (tenv : Typing_env.t) (t1 t2 : flambda_type),
    Typing_env.is_bottom tenv = true ->
    meet tenv t1 t2 = Some Or_bottom.Bottom.

Definition meet_contract (meet : meet_function) : Prop :=
  meet_sound meet
  ∧ meet_keeps_well_formed meet
  ∧ meet_kinds meet
  ∧ meet_bottom_env meet.

(* ======================================================================== *)
(* Adding to an environment                                                 *)
(* ======================================================================== *)

(* [Typing_env.add_equation]. If a valuation satisfies the environment,
   and the name has a value, of the type given, then a valuation that
   agrees with the first on what the environment knew satisfies the
   environment answered. In particular that environment is not
   bottom. *)
(* CR mshinwell: the premise is that the environment is satisfied
   strictly: the name, if it holds poison, has a type that is true of
   poison, and so has every name whose type the meet reads. That is an
   assumption of the compiler that nothing checks. Unboxing gives types
   to parameters that hold poison on some paths; an equation added on
   such a parameter where it holds poison, whose meet with its type is
   bottom, would make the environment of a point that is reached
   bottom. Decision D26. *)
Definition add_equation_sound (add_equation : add_equation_function)
    : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv tenv' : Typing_env.t)
    (n : name) (ty : flambda_type) (v : value),
    typing_env_fits tenv ->
    type_fits tenv ty ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation tenv ->
    value_of_name valuation n = Some v ->
    type_satisfied statics valuation v ty ->
    add_equation tenv n ty = Some tenv' ->
    ∃ valuation' : env,
      extended_from tenv valuation valuation'
      ∧ typing_env_satisfied_strictly statics valuation' tenv'
      ∧ keeps_equations_valued tenv tenv' valuation valuation'.

(* The same for [typing_env_satisfied], where no name holds poison. It
   follows from [add_equation_sound]
   ([add_equation_no_poison_statement]). *)
Definition add_equation_sound_no_poison
    (add_equation : add_equation_function) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv tenv' : Typing_env.t)
    (n : name) (ty : flambda_type) (v : value),
    typing_env_fits tenv ->
    type_fits tenv ty ->
    statics_valid statics ->
    typing_env_satisfied statics valuation tenv ->
    (∀ n : name, ~ holds_poison valuation n) ->
    value_of_name valuation n = Some v ->
    type_satisfied statics valuation v ty ->
    add_equation tenv n ty = Some tenv' ->
    ∃ valuation' : env,
      extended_from tenv valuation valuation'
      ∧ typing_env_satisfied_strictly statics valuation' tenv'
      ∧ keeps_equations_valued tenv tenv' valuation valuation'.

(* [Typing_env.add_env_extension]: the same for the equations of an
   extension. That the extension is satisfied says that each of its
   names has a value, of the type of its equation
   (FlambdaTypeMeaningSpec.v). *)
Definition add_env_extension_sound
    (add_env_extension : add_env_extension_function) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv tenv' : Typing_env.t)
    (extension : env_extension),
    typing_env_fits tenv ->
    env_extension_fits tenv extension ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_extension_satisfied statics valuation extension ->
    add_env_extension tenv extension = Some tenv' ->
    ∃ valuation' : env,
      extended_from tenv valuation valuation'
      ∧ typing_env_satisfied_strictly statics valuation' tenv'
      ∧ keeps_equations_valued tenv tenv' valuation valuation'.

(* [Typing_env.add_equations_on_params]: the same for the parameters of
   a continuation and the types of its arguments. Each parameter has a
   value, of the type given for it. *)
Definition add_equations_on_params_sound
    (add_equations_on_params : add_equations_on_params_function) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv tenv' : Typing_env.t)
    (params : list (variable * flambda_kind))
    (param_types : list flambda_type),
    typing_env_fits tenv ->
    (∀ ty : flambda_type, ty ∈ param_types -> type_fits tenv ty) ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation tenv ->
    Forall2
      (fun (param : variable * flambda_kind) (ty : flambda_type) =>
         ∃ v : value,
           lookup (fst param) valuation = Some v
           ∧ type_satisfied statics valuation v ty)
      params param_types ->
    add_equations_on_params tenv params param_types = Some tenv' ->
    ∃ valuation' : env,
      extended_from tenv valuation valuation'
      ∧ typing_env_satisfied_strictly statics valuation' tenv'
      ∧ keeps_equations_valued tenv tenv' valuation valuation'.

(* ======================================================================== *)
(* The meet of expanded types, and the join that meet is given              *)
(* ======================================================================== *)

(* The second environment has every name that the first has a type
   for, with its binding time and name mode and a type of the same
   kind. *)
Definition keeps_names (tenv tenv' : Typing_env.t) : Prop :=
  ∀ (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    ∃ ty' : flambda_type,
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv')
      = Some (ty', binding_time_and_mode)
      ∧ kind ty' = kind ty.

(* An environment of a meet after some of the meet: its typing
   environment has the same scope, the same earlier levels and the
   same symbols defined, every simple has the kinds that it had, the
   names it knew are known with their binding times and name modes and
   with types of the same kinds, the names that equations are being
   added to are the same, and no name to avoid is forgotten. (A symbol
   that is defined and has no type is of the kind of values; that it
   is given no type of another kind is what the kinds add to what is
   said of the names.) *)
Definition meet_env_extends (menv menv' : Meet_env.t) : Prop :=
  let tenv := Meet_env.typing_env menv in
  let tenv' := Meet_env.typing_env menv' in
  Typing_env.current_scope tenv' = Typing_env.current_scope tenv
  ∧ Typing_env.prev_levels tenv' = Typing_env.prev_levels tenv
  ∧ Typing_env.min_binding_time tenv' = Typing_env.min_binding_time tenv
  ∧ Typing_env.defined_symbols tenv' = Typing_env.defined_symbols tenv
  ∧ (∀ (s : simple) (kind : flambda_kind),
      simple_of_kind tenv s kind -> simple_of_kind tenv' s kind)
  ∧ Meet_env.adding_equations_for_names menv'
    = Meet_env.adding_equations_for_names menv
  ∧ (∀ x : variable,
      among x (Meet_env.names_to_avoid menv) = true ->
      among x (Meet_env.names_to_avoid menv') = true)
  ∧ keeps_names tenv tenv'.

(* [Meet_env.meet_expanded_head], the meet of two types that are not
   aliases: [meet_sound] for it, in the environment of a meet. With
   [meet_expanded_head_keeps] it is what the join is given, and what
   the join's contract supposes. *)
Definition meet_expanded_head_sound (meet_head : meet_expanded_head) : Prop :=
  ∀ (statics : static_env) (valuation : env) (menv : Meet_env.t)
    (expanded1 expanded2 : Expanded_type.t) (v : value)
    (result : Meet_result.t Expanded_type.t),
    let tenv := Meet_env.typing_env menv in
    meet_inputs_well_formed tenv (Expanded_type.to_type expanded1)
      (Expanded_type.to_type expanded2) ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation tenv ->
    type_satisfied statics valuation v (Expanded_type.to_type expanded1) ->
    type_satisfied statics valuation v (Expanded_type.to_type expanded2) ->
    meet_head menv expanded1 expanded2 = Some result ->
    ∃ (r : meet_return_value Expanded_type.t) (menv' : Meet_env.t)
      (valuation' : env),
      result = Meet_result.Ok r menv'
      ∧ extended_from tenv valuation valuation'
      ∧ typing_env_satisfied_strictly statics valuation'
          (Meet_env.typing_env menv')
      ∧ type_satisfied statics valuation' v
          (Expanded_type.to_type (extract_value r expanded1 expanded2))
      ∧ keeps_equations_valued tenv (Meet_env.typing_env menv') valuation
          valuation'.

(* The variables that a meet has defined are new. A variable that the
   second environment knows and the first does not was not among the
   names to avoid of the first, and is among those of the second. The
   join makes its variables outside the names to avoid and adds them;
   so the variables that the meets of two alternatives define, the
   second being given the names to avoid of the first, are different
   variables. And such a variable may not occur in terms: the second
   environment has it at a name mode that is not the normal one
   ([Typing_env.mem] with the normal name mode for the least). *)
Definition new_names_avoided (menv menv' : Meet_env.t) : Prop :=
  ∀ x : variable,
    Typing_env.mem None (Meet_env.typing_env menv') (Name_var x) = true ->
    Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = false ->
    among x (Meet_env.names_to_avoid menv) = false
    ∧ among x (Meet_env.names_to_avoid menv') = true
    ∧ Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env menv')
        (Name_var x)
      = false.

(* What the meet of expanded types keeps, whatever the values:
   [meet_keeps_well_formed] for it. The environment answered is the
   first extended, it fits, and the type answered fits it and is of the
   kind of the inputs. The type answered is not obviously bottom (a
   meet of types that have no value in common answers bottom, and no
   type), and it is not obviously unknown unless it is the second
   input, or both: [Meet_env] gives the type answered to a name, where
   it is the first input or a new type, and the typing environment is
   to be given neither kind of type there. And the variables that
   it has defined are new ([new_names_avoided]). No value and no
   valuation is supposed. *)
Definition meet_expanded_head_keeps (meet_head : meet_expanded_head) : Prop :=
  ∀ (menv menv' : Meet_env.t) (expanded1 expanded2 : Expanded_type.t)
    (r : meet_return_value Expanded_type.t),
    let tenv := Meet_env.typing_env menv in
    let answered :=
      Expanded_type.to_type (extract_value r expanded1 expanded2)
    in
    meet_inputs_well_formed tenv (Expanded_type.to_type expanded1)
      (Expanded_type.to_type expanded2) ->
    meet_head menv expanded1 expanded2 = Some (Meet_result.Ok r menv') ->
    meet_env_extends menv menv'
    ∧ typing_env_fits (Meet_env.typing_env menv')
    ∧ type_fits (Meet_env.typing_env menv') answered
    ∧ kind answered = kind (Expanded_type.to_type expanded1)
    ∧ is_obviously_bottom answered = false
    ∧ (is_obviously_unknown answered = true ->
       r = Right_input ∨ r = Both_inputs)
    ∧ new_names_avoided menv menv'.

(* What meet gives the join: an environment made from [initial_tenv] by
   adding definitions and equations. It fits, it has what
   [initial_tenv] has, and it knows no symbol that [initial_tenv] does
   not. It is not said that what satisfies it satisfies
   [initial_tenv]: see the header. *)
(* The last clause was added on 2026-11-12, after the contract was
   signed, at the request of the owner of the n-way join: its join
   copies an alias to a symbol into a type of the target, which fits
   the target only if the target knows the symbol. It is a premise of
   the join's contract, so it asks less of the join, and one thing
   more of the meet, which defines no symbol. *)
Definition scoped_env_of (initial_tenv scoped_env : Typing_env.t) : Prop :=
  typing_env_fits scoped_env
  ∧ (∀ n : name,
      Typing_env.mem None initial_tenv n = true ->
      Typing_env.mem None scoped_env n = true)
  ∧ keeps_names initial_tenv scoped_env
  ∧ ∀ sym : symbol,
      Typing_env.mem None scoped_env (Name_symbol sym) = true ->
      Typing_env.mem None initial_tenv (Name_symbol sym) = true.

(* No two of the environments that are joined have one [Simple] at
   two kinds. *)
Definition scoped_envs_agree (scoped_envs : list Typing_env.t) : Prop :=
  ∀ (scoped_env scoped_env' : Typing_env.t) (s : simple)
    (kind kind' : flambda_kind),
    scoped_env ∈ scoped_envs ->
    scoped_env' ∈ scoped_envs ->
    simple_of_kind scoped_env s kind ->
    simple_of_kind scoped_env' s kind' ->
    kind = kind'.

(* The contract of the join that meet is given. Meet calls it with the
   typing environment of [initial_env] for [initial_tenv] and its scope
   for [cut_after]. *)
Definition join_scoped_envs_sound (join : join_scoped_envs) : Prop :=
  ∀ (meet_head : meet_expanded_head) (initial_env result : Meet_env.t)
    (scoped_envs : list Typing_env.t),
    let initial_tenv := Meet_env.typing_env initial_env in
    meet_expanded_head_sound meet_head ->
    meet_expanded_head_keeps meet_head ->
    typing_env_fits initial_tenv ->
    (∀ scoped_env : Typing_env.t,
      scoped_env ∈ scoped_envs -> scoped_env_of initial_tenv scoped_env) ->
    scoped_envs_agree scoped_envs ->
    join meet_head (Typing_env.current_scope initial_tenv) initial_env
      initial_tenv scoped_envs
    = Some result ->
    (* What satisfies the first environment and one of the environments
       joined extends to something that satisfies the result. The
       equations that the environment joined has after the scope of
       the first are on names that have a value, and the join keeps
       that of the first environment: it adds no equation on a name
       that has no value. *)
    (∀ (scoped_env : Typing_env.t) (statics : static_env) (valuation : env),
      scoped_env ∈ scoped_envs ->
      statics_valid statics ->
      typing_env_satisfied_strictly statics valuation initial_tenv ->
      typing_env_satisfied_strictly statics valuation scoped_env ->
      equations_valued scoped_env (Typing_env.current_scope initial_tenv)
        valuation ->
      ∃ valuation' : env,
        extended_from scoped_env valuation valuation'
        ∧ typing_env_satisfied_strictly statics valuation'
            (Meet_env.typing_env result)
        ∧ keeps_equations_valued initial_tenv (Meet_env.typing_env result)
            valuation valuation')
    (* The result is the first environment, extended. *)
    ∧ meet_env_extends initial_env result
    ∧ typing_env_fits (Meet_env.typing_env result)
    (* A variable of the result that no environment given knows is new,
       and is to be avoided from now on. *)
    ∧ (∀ x : variable,
        Typing_env.mem None (Meet_env.typing_env result) (Name_var x)
        = true ->
        Typing_env.mem None initial_tenv (Name_var x) = false ->
        (∀ scoped_env : Typing_env.t,
          scoped_env ∈ scoped_envs ->
          Typing_env.mem None scoped_env (Name_var x) = false) ->
        among x (Meet_env.names_to_avoid initial_env) = false
        ∧ among x (Meet_env.names_to_avoid result) = true)
    (* A variable of the result that the first environment does not
       know may not occur in terms, whether or not an environment
       joined knows it. *)
    (* CR mshinwell: this clause was added on 2026-11-12, after the
       contract was signed, for the rewrite of the [Switch]: Simplify
       goes on, in the environment that a meet answers, with the
       terms that it had; a meet gives the machine no binding, and
       what satisfied the environment given is known to bind the
       variables that could occur in terms under it, not one that
       newly could ([meet_to_prove_keeps_terms_statement],
       FlambdaMeetTermsSpec.v). It asks one thing more of the join.
       The join that forgets has it, since it makes no variable. The
       compiler's join defines the variables of its target with the
       name mode of the variables that exist only in types
       ([define_or_eliminate_variables_and_add_equations] of
       types/env/join_env.ml, the two calls of
       [ME.add_variable_definition]), and so does the meet where it
       defines variables after a join ([add_defined_vars] and the
       fold over the extra variables in
       types/meet_and_n_way_join.ml). No place was found in types/
       where a meet or a join defines a variable with the normal name
       mode. *)
    ∧ (∀ x : variable,
        Typing_env.mem None (Meet_env.typing_env result) (Name_var x)
        = true ->
        Typing_env.mem None initial_tenv (Name_var x) = false ->
        Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env result)
          (Name_var x)
        = false)
    (* A [Simple] that the result knows is of the kind that an
       environment joined has it at. *)
    ∧ ∀ (scoped_env : Typing_env.t) (s : simple) (kind : flambda_kind),
        scoped_env ∈ scoped_envs ->
        Typing_env.mem_simple None (Meet_env.typing_env result) s = true ->
        simple_of_kind scoped_env s kind ->
        simple_of_kind (Meet_env.typing_env result) s kind.

(* What is to be proved of the transcribed meet, which takes the join
   and the fuel: with a join that has its contract, it has its own,
   whatever the fuel. *)
Definition transcription_sound
    (meet_of : join_scoped_envs -> nat -> meet_function) : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_sound join -> meet_contract (meet_of join fuel).

(* ======================================================================== *)
(* The transcribed meet                                                     *)
(* ======================================================================== *)

(* The meet as the compiler has it (FlambdaMeet.v, [compiler_choices]):
   the step for [Get_tag] adds [blocks_with_these_tags], the variables
   of a level are defined again after a join, and the variables of a
   single case of a row-like structure are not lifted. It has not
   [meet_sound] (FlambdaMeetExamplesSpec.v). It is not the compiler's
   meet on every input: where an equation is being added to a name,
   the meet in between has demoted the name, and the type answered
   has one constant for only value, the compiler goes on and this
   function answers [None] (FlambdaMeetEnv.v; discrepancy T4-10).
   There are such inputs among types built by hand with the compiler's
   library of types. Whether Simplify reaches one is not known. *)
Definition compiler_meet (join : join_scoped_envs) (fuel : nat)
    : meet_function :=
  Meet.meet join compiler_choices fuel.

(* The meet that the contract is to be proved of ([repaired_choices]):
   the step for [Get_tag] adds [immediate_or_blocks_with_these_tags],
   a variable that the environment knows is not defined again, and the
   variables of a single case are lifted. [transcription_sound
   meet_to_prove] is proved ([meet_to_prove_sound_statement],
   FlambdaMeetClosuresSpec.v). *)
Definition meet_to_prove (join : join_scoped_envs) (fuel : nat)
    : meet_function :=
  Meet.meet join repaired_choices fuel.

(* ======================================================================== *)
(* The step of the compiler's meet that is not sound                        *)
(* ======================================================================== *)

(* The compiler's step is not sound: there are a value that is not
   poison and a number that [Get_tag] relates, and the value has not
   the type that the step adds. The value is an immediate. *)
Definition get_tag_reduction_unsound_statement : Prop :=
  ∃ (statics : static_env) (vars : env) (w : value) (n : Z),
    relation_holds statics Relation.Get_tag w n
    ∧ ~ is_poison w
    ∧ ~ type_satisfied statics vars w
          (blocks_with_these_tags [n] Alloc_mode.Heap_or_local).

(* With the other type it is, for the tags that the step computes from
   the numbers: those that are tags of blocks ([is_tag], FlambdaMeet.v).
   The relation gives that the value is not poison, and that if it is
   a block its tag is in range (see the header). *)
Definition get_tag_reduction_repaired_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (w : value) (n : Z) (numbers : list Z)
    (alloc_mode : Alloc_mode.t),
    relation_holds statics Relation.Get_tag w n ->
    n ∈ numbers ->
    type_satisfied statics vars w
      (immediate_or_blocks_with_these_tags
         (Keyed.dedup Z.eqb (filter is_tag numbers)) alloc_mode).

(* ======================================================================== *)
(* What the definitions give                                                *)
(* ======================================================================== *)

(* A meet that has its contract answers bottom only where no value has
   both types. *)
Definition meet_answers_bottom_statement : Prop :=
  ∀ meet : meet_function, meet_sound meet -> meet_bottom_sound meet.

(* [add_equation_sound] gives the statement for
   [typing_env_satisfied] where no name holds poison. *)
Definition add_equation_no_poison_statement : Prop :=
  ∀ add_equation : add_equation_function,
    add_equation_sound add_equation ->
    add_equation_sound_no_poison add_equation.

(* Each type of an environment that fits, fits it. *)
Definition fits_types_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    typing_env_fits tenv ->
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    type_fits tenv ty.
