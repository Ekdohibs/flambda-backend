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

(* The rebuild with unboxing, in the mode of the fix and in that of its
   variant, refines its unit across calls: code whose parameters and
   results are unboxed, and code that keeps its calling convention, called
   directly, in tail position or not, with or without the wrapper of
   [make_apply_wrapper]; and calls whose code is not known, of unknown
   arity or of known arity, full calls, partial applications and
   over-applications, where every code of the unit keeps its calling
   convention.

   This file is to be reviewed: the fragment ([calls_unboxing]), the
   statements and the examples, against FlambdaReaperUnboxing.v and
   FlambdaReaperCallingConventionsSpec.v. The proofs are in
   FlambdaReaperUnboxingRefinesCallsProofs.v, which needs no review.

   [rebuild_unbox_fixed_refines_cc_statement]: for a unit [e] of the
   fragment of FlambdaReaperFlowSpec.v ([in_flow_fragment]), tables [T]
   closed under the rules ([check_closed]) and any table of
   [cannot_change_calling_convention] ([cannot_change]), the unit rebuilt
   with unboxing in the mode of the fix ([rebuild_unit_unbox_fixed]: no
   change of representation of closures) refines the unit rebuilt with
   the change of calling conventions alone, from the same table
   ([rebuild_unit_cc_from_tables]), run on its own ([program_refines]),
   if the unit is of [calls_unboxing] and the reads that the rebuild
   keeps of the names unboxed find their fields ([check_unboxed_reads]).
   The two rebuilt terms are the same but for the unboxing (as in
   FlambdaReaperUnboxingRefinesSpec.v), the parameters that the rebuild
   with unboxing keeps with no use ([Keep_unused]), to which it gives
   poison, the calls of code whose parameters or results are unboxed,
   which give the variables of the fields of their arguments and return
   those of their results, and the wrappers of the calls.

   [rebuild_unbox_fixed_refines_observably_statement]: with the premises
   of
   FlambdaReaperCallingConventionsSpec.rebuild_cc_refines_observably_statement,
   the rebuilt unit refines the unit up to observation. It is that
   statement composed with the first.

   [rebuild_unbox_fixed_tail_refines_cc_statement] and
   [rebuild_unbox_fixed_tail_refines_observably_statement]: the same of
   the variant of the fix ([rebuild_unit_unbox_fixed_tail]).

   The [_local] versions of the two statements up to observation hold
   with local fields: they ask the check of the fix of local fields
   ([check_local_reads_used], FlambdaReaperLocalFieldsSpec.v) in place
   of no local field or no projection, and compose the [_local]
   statement of FlambdaReaperCallingConventionsSpec.v with the first.
   The statements with the old premise are their corollaries.

   [make_apply_wrapper_u] gives poison to a parameter of the continuation
   of a call kept with no use ([Keep_unused]) whatever the callee does
   with the result. The variant (the mode [tail] of
   FlambdaReaperUnboxing.v) passes a result that the callee keeps through
   instead, so that the call needs no wrapper if that is all it would do,
   and gives poison only for a result that the callee unboxes or deletes.
   The two differ on a call whose continuation keeps a result with no use
   and whose callee keeps it: in this model, a call in tail position from
   code that keeps its calling convention and whose result has no use
   (the first parameter of an exception handler is the other parameter
   kept with no use, and a well-formed unit never makes such a handler the
   continuation of a call). The fix returns to a wrapper that gives
   poison, and the call is no longer in tail position; the variant leaves
   it in tail position. Both refine the unit: the relation lets the
   parameter be anything of its kind ([tail_example_statement]).

   [fixed_call_example_statement], [fixed_wrapper_example_statement]: the
   units of FlambdaReaperUnboxingRefinesSpec.CallExample (a call of code
   whose parameter and result are unboxed, with no wrapper) and
   WrapperExample (the same code called twice, once through a wrapper
   that drops the field of the result), with the tables of the solver of
   the test tool, are of the fragment, the rebuild in the mode of the fix
   makes the terms that the rebuild of the branch makes, and the rebuilt
   units refine the units up to observation.

   [tail_example_statement]: a unit whose code 51, which keeps its calling
   convention and whose result has no use, calls the code 50 in tail
   position, with the tables of the least solution: it is of the fragment,
   the variant of the fix rebuilds it as the change of calling conventions
   alone does, with the call in tail position, the fix makes the call
   return to a wrapper, and both rebuilt units refine the unit up to
   observation.

   [exn_example_statement]: a unit whose code 50, which keeps its calling
   convention, raises its parameter to a handler of the unit, which the
   unit installs and gives the call as its exception continuation with
   two extra arguments, with the tables of the least solution: it is of
   the fragment, the two rebuilds and the variant of the fix rebuild it
   alike, the extra argument whose parameter has no use dropped, and the
   rebuilt unit refines the unit up to observation.

   [poison_arg_example_statement]: a unit that gives the pair [x10], whose
   field it reads, to code that cannot change its calling convention (it
   escapes) and has no use of its parameter, with the tables of the least
   solution: the rules of the fix unbox [x10], the mask of the call keeps
   no argument, the unit is of the fragment, the rebuild in the mode of the
   fix gives the call poison, and the rebuilt unit refines the unit up to
   observation.

   [escaping_argument_calls_statement]: the unit of the fault of the
   compiler FlambdaReaperUnboxingSpec.escaping_argument_unit, which gives
   an unboxed argument to a parameter with a use of code that cannot
   change its calling convention, and calls that parameter, with the
   tables of the solver: with the rules of the fix nothing is unboxed and
   the unit is of the fragment, and the rebuilt unit refines it; with
   those of the branch the argument [x10] is unboxed, the mask of the call
   keeps it, and the unit is not of the fragment.

   The fragment ([calls_unboxing]), for the unboxed fields that the tables
   give ([unit_unboxed_fields]), the solution and the table:
   - The local unboxing of FlambdaReaperUnboxingRefinesSpec.v: an unboxed
     variable is bound to a block, a boxed number, an alias or the read of
     a field unboxed in turn ([unboxed_binding_ok]), or is a parameter,
     and is otherwise only read ([plain_binding_ok]) or given to a
     parameter unboxed alike ([arg_ok]). No kind is [Region] or [Rec_info]
     ([kind_ok]).
   - The continuations in scope are described by their parameters
     ([pdesc]): the decision of the rebuild with unboxing in the mode of
     the fix ([handler_decisions], or the decisions of the results of
     code), whether the change of calling conventions keeps it, and its
     kind ([pdesc_ok]: an unboxed parameter with fields is kept, and is a
     value); and by whether they may be returned to: a handler and the
     return continuation of code may be, the exception continuation of
     code is only raised to, by an [Apply_cont] or by a call
     ([apply_cont_ok2], [exn_cont_ok]).
   - Closures: a [Let] of a closure of a set with no value slot, whose
     other closures are not projected ([closure_binding_ok]).
   - Code ([calls_code]): code of the unit keeps its [my_closure], which
     is not unboxed, and changes its calling convention, or keeps it, is
     not tupled and unboxes no result ([code_pdescs]). It has as many
     parameters as its arity, each
     unboxed as the parameter of its code dep is, at the same paths, or
     not unboxed if the code keeps its calling convention
     ([code_param_ok]), and its body is of the fragment, with its return
     continuation described by the decisions of its results: those of
     the change, or those of a continuation of fixed arity ([Keep] or
     [Keep_unused]), which the first rebuild keeps with no mask. Its
     exception continuation, which the first rebuild keeps with no mask
     too, is described by the decision of a continuation of fixed arity
     ([code_exn_decision]): kept, with a use or with none, so that the
     code raises poison where the exception has no use; not unboxed.
   - Calls ([call_ok]) of code that is known: direct, of the code they
     are known to call ([code_id_actually_called]), at
     the arities of that code (any number of groups of parameters and of
     results), by a variable that is used and not unboxed, returning to a
     handler of the term or, in tail position, to the return continuation
     of the code they are in. The exception continuation of a call
     ([exn_cont_ok]) is a way out of the unit, to which the callee raises
     an exception with a use, or the exception continuation of the code
     the call is in, both with no extra argument; or a handler of the
     term, with an extra argument for each of its other parameters, given
     as [arg_ok] says. A way out, the exception continuation of code and
     the first parameter of the handler take the exception of the callee
     as the parameter of the continuation of a call takes a result
     ([exn_param_ok], [result_ok]): the first rebuild keeps them, they
     are not unboxed, and one with a use takes an exception that the
     callee raises with a use. Each argument is given to a parameter of
     code that changes its calling convention as [arg_ok] says; to code
     that keeps it, both rebuilds give the arguments that the callee uses
     as they are and the others as poison (FlambdaReaperRebuild.v,
     [rebuild_arguments]), so the two must read the same uses and an
     argument given as it is be a constant, a symbol or a name with a use,
     not unboxed; one given as poison may be an unboxed name
     ([kept_args_ok], [masked_args_ok]). Each result is given to a parameter of the
     continuation of the same kind ([result_ok]): one deleted or kept with
     no use, whatever the callee does with it; one kept, of a result that
     the callee keeps (with a use, if the callee keeps its calling
     convention) or deletes; one unboxed, of a result that the callee
     unboxes alike. Otherwise the call does
     not return in the rebuild with unboxing ([wrapper_args_u] fails: the
     known faults of the branch, which the fix keeps from happening by
     the rules). A result that the callee unboxes with no field is,
     besides, given to a parameter deleted only if the change of calling
     conventions deletes the result too, and to one unboxed only if that
     change keeps the result and the parameter alike: otherwise it makes
     a wrapper where the rebuild with unboxing does not, which the proof
     does not relate.
   - Calls whose code is not known ([ind_call_ok]): of unknown arity, or
     of known arity with no one code id that the solution gives, by a
     variable that is used and not unboxed. Both rebuilds give them the
     arguments at the mask of the call ([call_mask], [masked_args_ok]),
     and expect every result, as calls of code that keeps its calling
     convention. A call may call the code of any closure, so every code of
     the unit must be one it may call ([ind_code_ok]): it keeps its
     calling convention; its results, where they are of the return arity
     of the call, are given to the parameters of its continuation as
     those of code called directly are; and a single result of kind
     [Value], which an over-application calls, has a use. The
     continuation takes the results as [ind_results] are given (none
     unboxed), and the exception continuation the exception of
     [ind_exn_decision]: with no use if some code of the unit raises with
     none.
   - [check_unboxed_uses]: no use of an unboxed name is called or has a
     function slot read from it, so that [add_usages_through_function_slots]
     adds no use to the fields of an unboxed name. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperLeast FlambdaReaperLocalFields.
From OxCamlFlambda2 Require FlambdaReaperTupledSpec
  FlambdaReaperCallingConventionsSpec FlambdaReaperUnboxingSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* A parameter of a continuation: its decision in the rebuild with
   unboxing, in the mode of the fix, whether the change of calling
   conventions keeps it, and its kind. *)
Record pdesc : Type := Pdesc {
  pd_decision : decision;
  pd_keep : bool;
  pd_kind : flambda_kind
}.

(* The paths of the leaves of unboxed fields, in the order of
   [unboxed_field_leaves]. *)
Fixpoint ufield_paths (kind : flambda_kind) (u : unboxed_field) {struct u}
    : list (list field * flambda_kind) :=
  match u with
  | Not_unboxed _ => [([], kind)]
  | Unboxed fields =>
    (fix go (fields : list (field * unboxed_field))
         : list (list field * flambda_kind) :=
       match fields with
       | [] => []
       | (f, u) :: fields =>
         map (fun pk => (f :: fst pk, snd pk)) (ufield_paths (field_kind f) u)
         ++ go fields
       end) fields
  end.

Definition fields_paths (fields : unboxed_fields)
    : list (list field * flambda_kind) :=
  ufield_paths Value (Unboxed fields).

Fixpoint fields_list_eqb (p p' : list field) : bool :=
  match p, p' with
  | [], [] => true
  | f :: p, f' :: p' => field_eqb f f' && fields_list_eqb p p'
  | _, _ => false
  end.

Fixpoint paths_eqb (l l' : list (list field * flambda_kind)) : bool :=
  match l, l' with
  | [], [] => true
  | (p, k) :: l, (p', k') :: l' =>
    fields_list_eqb p p' && kind_eqb k k' && paths_eqb l l'
  | _, _ => false
  end.

(* Two unboxings of the same shape: their leaves at the same paths, in
   the same order. *)
Definition same_paths (fields fields' : unboxed_fields) : bool :=
  paths_eqb (fields_paths fields) (fields_paths fields').

(* An unboxing with no variable, at any depth. *)
Definition no_leaves (fields : unboxed_fields) : bool :=
  match fields_leaves fields with [] => true | _ :: _ => false end.

Definition kind_ok (kind : flambda_kind) : bool :=
  match kind with
  | Value | Naked_number _ => true
  | Region | Rec_info => false
  end.

Fixpoint forallb2 {A B : Type} (p : A -> B -> bool) (l : list A)
    (l' : list B) : bool :=
  match l, l' with
  | [], [] => true
  | a :: l, b :: l' => p a b && forallb2 p l l'
  | _, _ => false
  end.

Section Fragment.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).

(* The decision of a parameter agrees with what the change of calling
   conventions keeps and with its kind: a parameter unboxed to some
   variable is kept, and is a value. *)
Definition pdesc_ok (p : pdesc) : bool :=
  kind_ok (pd_kind p)
  && match pd_decision p with
     | Decided (Keep kind) | Keep_unused kind =>
       pd_keep p && kind_eqb kind (pd_kind p)
     | Decided Delete => negb (pd_keep p)
     | Unbox fields =>
       kind_eqb (pd_kind p) Value && (pd_keep p || no_leaves fields)
     end.

(* The argument [a] of a parameter [p]: unboxed alike, and used if the
   change of calling conventions keeps the parameter, for an unboxed
   parameter; not an unboxed name for a parameter kept; anything for a
   parameter deleted or kept with no use. *)
Definition arg_ok (kinds : finite_map flambda_kind) (p : pdesc) (a : simple)
    : bool :=
  match pd_decision p with
  | Unbox fields => unboxed_arg_ok unboxed sol kinds (pd_keep p) fields a
  | Decided (Keep _) => plain_simple unboxed a
  | Decided Delete | Keep_unused _ => true
  end.

Fixpoint args_ok2 (kinds : finite_map flambda_kind) (ps : list pdesc)
    (args : list simple) : bool :=
  match ps, args with
  | [], _ => forallb (plain_simple unboxed) args
  | _ :: _, [] => true
  | p :: ps, a :: args => arg_ok kinds p a && args_ok2 kinds ps args
  end.

(* [C] describes the continuations in scope, each with whether it may be
   returned to: a handler of the term or the return continuation of code
   may be, the exception continuation of code is only raised to, by an
   [Apply_cont] or by the calls in its body. *)
Definition apply_cont_ok2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ac : apply_cont) : bool :=
  match lookup (ac_continuation ac) C with
  | Some (ps, _) => args_ok2 kinds ps (ac_args ac)
  | None => forallb (plain_simple unboxed) (ac_args ac)
  end.

(* A way out of the unit, as the exception continuation of a call: it
   takes the exception as a parameter of fixed arity with a use. *)
Definition exn_pdescs : list pdesc := [Pdesc (Decided (Keep Value)) true Value].

(* The decision of the parameter of the exception continuation of code
   [id] in the rebuild with unboxing ([code_continuation_decisions]), of a
   continuation of fixed arity: kept, with a use or not. The first rebuild
   keeps it as it is. [None] where it is unboxed. *)
Definition code_exn_decision (id : code_id) : option decision :=
  match lookup id code_deps with
  | Some cd =>
    match fixed_arity_decisions sol unboxed [cd_exn cd] [Value] with
    | [Unbox _] => None
    | [d] => Some d
    | _ => None
    end
  | None => None
  end.

(* The parameters of a handler, as the two rebuilds have them. *)
Definition handler_pdescs (is_exn : bool) (h : continuation_handler)
    : list pdesc :=
  match h with
  | Handler _ metadata _ =>
    map (fun '(d, keep, kind) => Pdesc d keep kind)
      (combine
         (combine (handler_decisions true sol unboxed is_exn h)
            (handler_mask sol is_exn h))
         (hm_params_kinds metadata))
  end.

Fixpoint pdescs_of (ds : list decision) (keeps : list bool)
    (kinds : list flambda_kind) : list pdesc :=
  match ds, keeps, kinds with
  | d :: ds, keep :: keeps, kind :: kinds =>
    Pdesc d keep kind :: pdescs_of ds keeps kinds
  | _, _, _ => []
  end.

Definition is_unbox (d : decision) : bool :=
  match d with
  | Unbox _ => true
  | Decided _ | Keep_unused _ => false
  end.

(* The parameters and the results of code [id], as the two rebuilds have
   them: for code that changes its calling convention, its decisions; for
   code that keeps it, every parameter kept, and the decisions of its
   return continuation, which has a fixed arity: none unboxes a result (a
   caller would not expect it), and the code is not tupled. *)
Definition code_pdescs (id : code_id) (metadata : code_metadata)
    : option (list pdesc * list pdesc) :=
  match
    code_change_u sol cannot_change code_deps unit_code_ids unboxed id
      metadata,
    code_calling_convention_change sol cannot_change code_deps unit_code_ids
      id metadata
  with
  | Changing_u pd rd, Changing_calling_convention pd_cc rd_cc =>
    Some (pdescs_of pd (map is_kept pd_cc) (unarize (cm_params_arity metadata)),
          pdescs_of rd (map is_kept rd_cc)
            (unarize (cm_result_arity metadata)))
  | Not_changing_u, Not_changing_calling_convention =>
    match lookup id code_deps with
    | Some cd =>
      let kinds := unarize (cm_result_arity metadata) in
      let rd := fixed_arity_decisions sol unboxed (cd_return cd) kinds in
      if cm_is_tupled metadata || existsb is_unbox rd
      then None
      else
        Some (map (fun kind => Pdesc (Decided (Keep kind)) true kind)
                (unarize (cm_params_arity metadata)),
              pdescs_of rd (map (fun _ => true) rd) kinds)
    | None => None
    end
  | _, _ => None
  end.

(* The arguments [args] of a call that both rebuilds give as they are
   where the mask of the first, [mask], keeps them and as poison
   elsewhere ([FlambdaReaperRebuild.rebuild_arguments]): the second reads
   the same mask, [mask_u], with the names unboxed taken to have no use;
   an argument given as it is is not an unboxed name, and is a constant,
   a symbol or a name with a use; one given as poison may be anything, an
   unboxed name among others, which both give as the poison of its
   kind. *)
Definition masked_args_ok (mask mask_u : list bool) (args : list simple)
    : bool :=
  forallb2 Bool.eqb mask mask_u
  && forallb2
       (fun keep a =>
          negb keep
          || plain_simple unboxed a
             && match a with
                | Var z => s_has_usage sol (Node_variable z)
                | Symbol _ | Const _ => true
                end)
       mask args.

(* The arguments of a direct call of [f], of code that keeps its calling
   convention, at the mask of a call of known arity. *)
Definition kept_args_ok (f : variable) (args : list simple) : bool :=
  masked_args_ok
    (arguments_used_by_known_arity_call F sol (Node_variable f) args)
    (arguments_used_by_known_arity_call F
       (solution_without_unboxed sol unboxed) (Node_variable f) args)
    args.

(* The mask of the arguments of a call [ap] of [f] whose code is not known
   ([rebuild_arguments]), read off the solution [s]. *)
Definition call_mask (s : solution) (f : variable) (ap : apply_expr)
    : list bool :=
  match ap_call_kind ap with
  | Direct _ =>
    arguments_used_by_known_arity_call F s (Node_variable f) (ap_args ap)
  | Indirect =>
    arguments_used_by_unknown_arity_call F s (Node_variable f)
      (ap_args_arity ap)
  end.

(* The parameter [k] of the handler that a call returns to, given the
   result [r] of the callee. *)
Definition result_ok (k r : pdesc) : bool :=
  kind_eqb (pd_kind k) (pd_kind r)
  && match pd_decision k, pd_decision r with
     | Decided Delete, Unbox fields_r =>
       negb (no_leaves fields_r && pd_keep r)
     | Decided Delete, _ | Keep_unused _, _ => true
     | Decided (Keep _), Decided _ => true
     | Unbox fields_k, Unbox fields_r =>
       fields_alike fields_k fields_r
       && (negb (no_leaves fields_r) || Bool.eqb (pd_keep k) (pd_keep r))
     | _, _ => false
     end.

Definition is_none {A : Type} (o : option A) : bool :=
  match o with None => true | Some _ => false end.

(* The parameter [k] that takes the exception [r] of a callee: kept by the
   first rebuild, not unboxed, and given the exception as the parameter of
   the continuation of a call is given a result ([result_ok]), so that an
   exception that the callee raises with no use reaches only a parameter
   with no use. *)
Definition exn_param_ok (k r : pdesc) : bool :=
  pd_keep k && negb (is_unbox (pd_decision k)) && result_ok k r.

(* The exception continuation [e] of a call, with the extra arguments
   [extra], of code whose exception continuation is described by [r]: a
   way out of the unit, which takes the exception as it is
   ([exn_pdescs]); the exception continuation of the code the call is in;
   or a handler of the term, whose first parameter takes the exception
   and whose others the extra arguments, one each ([arg_ok]). There is no
   extra argument but for a handler. *)
Definition exn_cont_ok (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (e : continuation) (r : pdesc)
    (extra : list (simple * flambda_kind)) : bool :=
  match lookup e C with
  | None =>
    forallb2 exn_param_ok exn_pdescs [r]
    && match extra with [] => true | _ :: _ => false end
  | Some ([k], false) =>
    exn_param_ok k r && match extra with [] => true | _ :: _ => false end
  | Some (k :: ps, true) =>
    exn_param_ok k r && Nat.eqb (length extra) (length ps)
    && args_ok2 kinds ps (map fst extra)
  | Some _ => false
  end.

(* A call whose code is not known, of unknown arity or of known arity
   but with no one code id that the solution gives, may call the code of
   any closure: both rebuilds give it the arguments that the mask keeps
   and expect every result. Its results, as the second run takes them:
   kept, with a use where the continuation [kps] keeps the result with a
   use, and with none elsewhere ([kinds] are those of the results). *)
Fixpoint ind_results (kps : list pdesc) (kinds : list flambda_kind)
    : list pdesc :=
  match kps, kinds with
  | kp :: kps, kind :: kinds =>
    Pdesc
      (match pd_decision kp with
       | Decided (Keep _) => Decided (Keep kind)
       | Decided Delete | Unbox _ | Keep_unused _ => Keep_unused kind
       end)
      true kind
    :: ind_results kps kinds
  | _, _ => []
  end.

Definition is_kept_used (d : decision) : bool :=
  match d with
  | Decided (Keep _) => true
  | Decided Delete | Unbox _ | Keep_unused _ => false
  end.

(* Its exception: with a use, unless some code of the unit raises with
   none. *)
Definition ind_exn_decision : decision :=
  if forallb
       (fun '(id, _) =>
          match code_exn_decision id with
          | Some d => is_kept_used d
          | None => true
          end)
       codes
  then Decided (Keep Value)
  else Keep_unused Value.

(* The code [id] of the unit, which a call whose code is not known may
   call, returning to the parameters [kps] at the return arity [RA]: it
   keeps its calling convention, so that the two rebuilds call it alike;
   its results, where they are of the arity of the call, are given to
   [kps] as those of code called directly are ([result_ok]); and a single
   result of kind [Value], which an over-application calls, has a use. *)
Definition ind_code_ok (kps : list pdesc) (RA : arity) (id : code_id)
    (md : code_metadata) : bool :=
  match
    code_change_u sol cannot_change code_deps unit_code_ids unboxed id md,
    code_pdescs id md, code_exn_decision id
  with
  | Not_changing_u, Some (_, rds), Some _ =>
    Nat.eqb (length rds) (length (unarize (cm_result_arity md)))
    && (negb (arity_eqb (cm_result_arity md) RA)
        || forallb2 result_ok kps rds)
    && (negb (arity_eqb (cm_result_arity md) [Singleton Value])
        || forallb (fun r => is_kept_used (pd_decision r)) rds)
  | _, _, _ => false
  end.

(* A call [ap] of [f] whose code is not known: [f] is used and not
   unboxed; the arguments are given at the mask of the call
   ([masked_args_ok]); the call returns to a handler of the term or, in
   tail position, to the return continuation of the code it is in, which
   takes the results as [ind_results] are given; its exception
   continuation takes the exception of [ind_exn_decision]; and every code
   of the unit is one it may call ([ind_code_ok]). *)
Definition ind_call_ok (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable)
    : bool :=
  plain_simple unboxed (Var f) && is_var_used sol f Value
  && Nat.eqb (length (ap_args ap)) (length (unarize (ap_args_arity ap)))
  && masked_args_ok (call_mask sol f ap)
       (call_mask (solution_without_unboxed sol unboxed) f ap) (ap_args ap)
  && exn_cont_ok kinds C (ap_exn_continuation ap)
       (Pdesc ind_exn_decision true Value) (ap_exn_extra_args ap)
  && match lookup (ap_continuation ap) C with
     | Some (kps, true) =>
       let kinds_r := unarize (ap_return_arity ap) in
       Nat.eqb (length kps) (length kinds_r)
       && forallb2 result_ok kps (ind_results kps kinds_r)
       && forallb (fun '(id, md) => ind_code_ok kps (ap_return_arity ap) id md)
            codes
     | _ => false
     end.

Definition call_ok (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) : bool :=
  match ap_call_kind ap, ap_callee ap, code_id_actually_called F sol ap with
  | Direct id, Var f, Some id' =>
    Pos.eqb id id' && plain_simple unboxed (Var f) && is_var_used sol f Value
    && match lookup id codes with
       | Some metadata =>
         match code_pdescs id metadata with
         | Some (pds, rds) =>
           arity_eqb (ap_args_arity ap) (cm_params_arity metadata)
           && arity_eqb (ap_return_arity ap) (cm_result_arity metadata)
           && is_none
                (my_closure_decision_u cannot_change code_deps unit_code_ids
                   unboxed id)
           && Nat.eqb (length (ap_args ap)) (length pds)
           && Nat.eqb (length pds)
                (length (unarize (cm_params_arity metadata)))
           && Nat.eqb (length rds)
                (length (unarize (cm_result_arity metadata)))
           && forallb pdesc_ok pds && forallb pdesc_ok rds
           && match
                code_change_u sol cannot_change code_deps unit_code_ids
                  unboxed id metadata
              with
              | Not_changing_u => true
              | Changing_u _ _ => args_ok2 kinds pds (ap_args ap)
              end
           && match code_exn_decision id with
              | Some d =>
                exn_cont_ok kinds C (ap_exn_continuation ap)
                  (Pdesc d true Value) (ap_exn_extra_args ap)
              | None => false
              end
           && match
                code_change_u sol cannot_change code_deps unit_code_ids
                  unboxed id metadata
              with
              | Not_changing_u => kept_args_ok f (ap_args ap)
              | Changing_u _ _ => true
              end
           && match lookup (ap_continuation ap) C with
              | Some (kps, true) => forallb2 result_ok kps rds
              | _ => false
              end
         | None => false
         end
       | None => false
       end
  | _, Var f, None => ind_call_ok kinds C ap f
  | _, _, _ => false
  end.

(* The [Let] of a closure, not unboxed: of a set with no value slot,
   whose other closures are not projected. *)
Definition closure_binding_ok (x : variable) (d : named) (body : expr)
    : bool :=
  match d with
  | Closure_of_set slot decls [] =>
    negb (is_none (lookup slot decls))
    && match projections_of_set x d body with [] => true | _ :: _ => false end
  | _ => false
  end.

(* A parameter [x] of code, whose code dep's parameter is [p]: unboxed at
   the same paths if [p] is, otherwise not unboxed. *)
Definition code_param_ok (x : variable) (p : pdesc) : bool :=
  match pd_decision p with
  | Unbox fields =>
    match fields_of unboxed x with
    | Some fields_x => same_paths fields fields_x
    | None => false
    end
  | Decided _ | Keep_unused _ => plain_simple unboxed (Var x)
  end.

Fixpoint calls_expr (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok unboxed sol kinds x kind d fields
       | None =>
         plain_binding_ok unboxed x kind d || closure_binding_ok x d body
       end
    && calls_expr ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb calls_binding bindings && calls_expr kinds C body
  | Let_cont_non_recursive k h body =>
    let ps := handler_pdescs (handler_is_exn_handler h) h in
    calls_handler kinds C ps h && calls_expr kinds ((k, (ps, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map (fun '(k, h) => (k, (handler_pdescs false h, true))) handlers ++ C
    in
    forallb
      (fun '(_, h) => calls_handler kinds C' (handler_pdescs false h) h)
      handlers
    && calls_expr kinds C' body
  | Apply ap => call_ok kinds C ap
  | Apply_cont ac => apply_cont_ok2 kinds C ac
  | Switch scrutinee arms =>
    plain_simple unboxed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok2 kinds C ac) arms
  | Invalid => true
  end
with calls_handler (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ps : list pdesc)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb pdesc_ok ps
    && calls_expr (rev (combine params (hm_params_kinds metadata)) ++ kinds) C
         handler
  end
with calls_binding (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => calls_code id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ _ => local_binding unboxed binding
  end
with calls_code (id : code_id) (c : code) {struct c} : bool :=
  match c with
  | Code params my_closure ret exn metadata body =>
    match lookup id codes with
    | Some metadata' => code_metadata_eqb metadata metadata'
    | None => false
    end
    && match code_pdescs id metadata with
       | Some (pds, rds) =>
         Nat.eqb (length params) (length pds)
         && Nat.eqb (length pds) (length (unarize (cm_params_arity metadata)))
         && Nat.eqb (length rds) (length (unarize (cm_result_arity metadata)))
         && forallb pdesc_ok pds && forallb pdesc_ok rds
         && forallb2 code_param_ok params pds
         && is_none
              (my_closure_decision_u cannot_change code_deps unit_code_ids
                 unboxed id)
         && plain_simple unboxed (Var my_closure)
         && negb (Pos.eqb ret exn)
         && match code_exn_decision id with
            | Some d =>
              calls_expr
                (rev ((my_closure, Value)
                      :: with_kinds params
                           (unarize (cm_params_arity metadata))))
                [(ret, (rds, true)); (exn, ([Pdesc d true Value], false))]
                body
            | None => false
            end
       | None => false
       end
  end.

End Fragment.

(* No use of an unboxed name is read but as a field of a block or of a
   boxed number: none is called, and no function slot is read from
   one. *)
Definition check_unboxed_uses (F : facts) (T : tables)
    (unboxed : list (node * unboxed_fields)) : bool :=
  forallb
    (fun '(_, f, base) =>
       is_block_like_field f
       || forallb (fun '(x, _) => negb (mem_pair x base (t_usages T))) unboxed)
    (f_accessor F).

(* The unit [e] is of the fragment, for the unboxing that the tables give
   from its graph [F], and the table [cannot_change]. *)
Definition calls_unboxing (F : facts) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  calls_expr F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    [] [] e
  && check_unboxed_uses F T unboxed.

(* The rebuild with unboxing in the mode of the fix, with no change of
   representation. *)
Definition rebuild_unit_unbox_fixed (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (e : expr) : expr :=
  rebuild_unit_unbox_repr_fixed (fun _ => false) F T cannot_change code_deps
    dominated to_unbox [] preserve_direct_calls e.

(* The same, in the variant of the fix that passes on a result that the
   callee keeps to a parameter kept with no use. *)
Definition rebuild_unit_unbox_fixed_tail (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (e : expr) : expr :=
  rebuild_unit_unbox_repr_fixed_tail (fun _ => false) F T cannot_change
    code_deps dominated to_unbox [] preserve_direct_calls e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_fixed_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_fixed_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too (FlambdaReaperLocalFieldsSpec.v). *)
Definition rebuild_unbox_fixed_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_fixed_tail_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_fixed_tail (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_fixed_tail_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed_tail (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too (FlambdaReaperLocalFieldsSpec.v). *)
Definition rebuild_unbox_fixed_tail_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed_tail (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Definition fixed_call_example_statement : Prop :=
  calls_unboxing CallExample.facts CallExample.tables
    CallExample.cannot_change CallExample.code_deps CallExample.dominated
    CallExample.to_unbox CallExample.unit_expr
  = true
  ∧ check_unboxed_reads CallExample.facts CallExample.tables
      CallExample.dominated CallExample.to_unbox
    = true
  ∧ rebuild_unit_unbox_fixed CallExample.facts CallExample.tables
      CallExample.cannot_change CallExample.code_deps CallExample.dominated
      CallExample.to_unbox true CallExample.unit_expr
    = CallExample.rebuilt
  ∧ program_refines_observably CallExample.unit_expr CallExample.rebuilt.

Definition fixed_wrapper_example_statement : Prop :=
  calls_unboxing WrapperExample.facts WrapperExample.tables
    WrapperExample.cannot_change WrapperExample.code_deps
    WrapperExample.dominated WrapperExample.to_unbox WrapperExample.unit_expr
  = true
  ∧ check_unboxed_reads WrapperExample.facts WrapperExample.tables
      WrapperExample.dominated WrapperExample.to_unbox
    = true
  ∧ rebuild_unit_unbox_fixed WrapperExample.facts WrapperExample.tables
      WrapperExample.cannot_change WrapperExample.code_deps
      WrapperExample.dominated WrapperExample.to_unbox true
      WrapperExample.unit_expr
    = WrapperExample.rebuilt
  ∧ program_refines_observably WrapperExample.unit_expr
      WrapperExample.rebuilt.

(* A tail call from code that keeps its calling convention and whose
   result has no use: the code 51 calls the code 50, which it is given
   the closure of, in tail position, and the unit ignores what 51
   returns. Neither code changes its calling convention (both are in the
   table), so the return continuation of 51 is kept with no use
   ([Keep_unused]). The tables are the least solution of the rules
   (FlambdaReaperLeast.compute_least); nothing is unboxed. *)
Module TailExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition call (callee : variable) (id : code_id) (args : list simple)
    (k exn : continuation) : apply_expr :=
  {| ap_callee := Var callee; ap_call_kind := Direct id; ap_args := args;
     ap_continuation := k; ap_exn_continuation := exn;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition metadata : code_metadata :=
  make_code_metadata [Singleton Value] [Singleton Value].

(* The code 50 returns its parameter; the code 51 calls 50 through its
   parameter [x44] in tail position. *)
Definition unit_with (body_50 body_51 : expr) (body : expr) : expr :=
  Let_static
    [Bind_code 50 (Code [40] 41 3 4 metadata body_50);
     Bind_code 51 (Code [44] 45 5 6 metadata body_51)]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Closure_of_set 71 [(71, 51)] []) body)).

Definition unit_expr : expr :=
  unit_with (return_to 3 [Var 40]) (Apply (call 44 50 [int 5] 5 6))
    (Let_cont_non_recursive 20
       (Handler [30] (make_handler_metadata [Value]) (return_to 1 [int 0]))
       (Apply (call 61 51 [Var 60] 20 2))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node := [c 50; c 51].

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* The rebuilt unit, the body of 51 being [body_51]: the result of 50,
   which has no use, is poison, and the call of 51 returns to a wrapper
   that drops its result. *)
Definition rebuilt_with (body_51 : expr) : expr :=
  unit_with (return_to 3 [Const (Poison Value)]) body_51
    (Let_cont_non_recursive 20
       (Handler [] (make_handler_metadata []) (return_to 1 [int 0]))
       (Let_cont_non_recursive 62
          (Handler [62] (make_handler_metadata [Value]) (return_to 20 []))
          (Apply (call 61 51 [Var 60] 62 2)))).

(* With the change of calling conventions alone, and in the variant of
   the fix, the call in 51 stays in tail position. *)
Definition rebuilt : expr := rebuilt_with (Apply (call 44 50 [int 5] 5 6)).

(* With the fix, it returns to a wrapper that gives poison to the return
   continuation of 51. *)
Definition rebuilt_fixed : expr :=
  rebuilt_with
    (Let_cont_non_recursive 62
       (Handler [62] (make_handler_metadata [Value])
          (return_to 5 [Const (Poison Value)]))
       (Apply (call 44 50 [int 5] 62 6))).

End TailExample.

Definition tail_example_statement : Prop :=
  calls_unboxing TailExample.facts TailExample.tables
    TailExample.cannot_change TailExample.code_deps [] []
    TailExample.unit_expr
  = true
  ∧ check_unboxed_reads TailExample.facts TailExample.tables [] [] = true
  ∧ rebuild_unit_cc_from_tables TailExample.facts TailExample.tables
      TailExample.cannot_change TailExample.code_deps true
      TailExample.unit_expr
    = TailExample.rebuilt
  ∧ rebuild_unit_unbox_fixed TailExample.facts TailExample.tables
      TailExample.cannot_change TailExample.code_deps [] [] true
      TailExample.unit_expr
    = TailExample.rebuilt_fixed
  ∧ rebuild_unit_unbox_fixed_tail TailExample.facts TailExample.tables
      TailExample.cannot_change TailExample.code_deps [] [] true
      TailExample.unit_expr
    = TailExample.rebuilt
  ∧ program_refines_observably TailExample.unit_expr TailExample.rebuilt_fixed
  ∧ program_refines_observably TailExample.unit_expr TailExample.rebuilt.

(* An exception raised by code to a handler of the unit, with extra
   arguments. The code 50, which keeps its calling convention (it is in
   the table), raises its parameter. The unit installs the handler 20
   and calls 50 with it as the exception continuation, with the extra
   arguments 7 and 8 for its parameters after the exception; the handler
   returns the first, and the second has no use. The tables are the
   least solution of the rules (FlambdaReaperLeast.compute_least);
   nothing is unboxed. *)
Module ExnExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition call (callee : variable) (id : code_id) (args : list simple)
    (k exn : continuation) (extra : list (simple * flambda_kind))
    : apply_expr :=
  {| ap_callee := Var callee; ap_call_kind := Direct id; ap_args := args;
     ap_continuation := k; ap_exn_continuation := exn;
     ap_exn_extra_args := extra;
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition jump (k : continuation) (args : list simple)
    (trap : option trap_action) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := trap |}.

(* The unit, the handler of the exception being [handler] with the
   parameters [params], the extra arguments of the call [extra], and the
   handler of the continuation of the call [return_handler]. *)
Definition unit_with (params : list variable) (handler : expr)
    (extra : list (simple * flambda_kind)) (return_handler : expr) : expr :=
  Let_static
    [Bind_code 50
       (Code [40] 41 3 4
          (make_code_metadata [Singleton Value] [Singleton Value])
          (jump 4 [Var 40] (Some (Pop 4 None))))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler params
             (make_exn_handler_metadata (map (fun _ => Value) params))
             handler)
          (Let_cont_non_recursive 21
             (Handler [32] (make_handler_metadata [Value]) return_handler)
             (Let_cont_non_recursive 22
                (Handler [] (make_handler_metadata [])
                   (Apply (call 60 50 [int 5] 21 20 extra)))
                (jump 22 [] (Some (Push 20))))))).

Definition unit_expr : expr :=
  unit_with [30; 31; 33] (return_to 1 [Var 31])
    [(int 7, Value); (int 8, Value)]
    (jump 1 [Var 32] (Some (Pop 20 None))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node := [c 50].

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* The rebuilt unit: the parameter 33 and the extra argument 8 are
   dropped, and the continuation 21, which nothing returns to, is
   [Invalid]. *)
Definition rebuilt : expr :=
  unit_with [30; 31] (return_to 1 [Var 31]) [(int 7, Value)] Invalid.

End ExnExample.

Definition exn_example_statement : Prop :=
  calls_unboxing ExnExample.facts ExnExample.tables ExnExample.cannot_change
    ExnExample.code_deps [] [] ExnExample.unit_expr
  = true
  ∧ check_unboxed_reads ExnExample.facts ExnExample.tables [] [] = true
  ∧ rebuild_unit_cc_from_tables ExnExample.facts ExnExample.tables
      ExnExample.cannot_change ExnExample.code_deps true ExnExample.unit_expr
    = ExnExample.rebuilt
  ∧ rebuild_unit_unbox_fixed ExnExample.facts ExnExample.tables
      ExnExample.cannot_change ExnExample.code_deps [] [] true
      ExnExample.unit_expr
    = ExnExample.rebuilt
  ∧ rebuild_unit_unbox_fixed_tail ExnExample.facts ExnExample.tables
      ExnExample.cannot_change ExnExample.code_deps [] [] true
      ExnExample.unit_expr
    = ExnExample.rebuilt
  ∧ program_refines_observably ExnExample.unit_expr ExnExample.rebuilt.

(* An unboxed argument given to code that keeps its calling convention,
   at a parameter with no use. The code 50 escapes (its closure [x60] is
   in the block that the unit returns), so that it cannot change its
   calling convention, and has no use of its parameter [x40]. The unit
   allocates the pair [x10], reads its first field into [x11], which it
   returns, and calls 50 with [x10]. The tables are the least solution of
   the rules; the rules of the fix unbox [x10]. The mask of the call keeps
   no argument, so that both rebuilds give the call poison, and the rebuild
   with unboxing binds [x11] to the variable of the field. *)
Module PoisonArgExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_static
    [Bind_code 50
       (Code [40] 41 3 4
          (make_code_metadata [Singleton Value] [Singleton Value])
          (return_to 3 [int 0]))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 10 Value (make_block [int 5; int 6])
          (Let 11 Value (load 0 (Var 10))
             (Let_cont_non_recursive 20
                (Handler [30] (make_handler_metadata [Value])
                   (Let 31 Value (make_block [Var 60; Var 11])
                      (return_to 1 [Var 31])))
                (Apply (call_50 [Var 10] 20)))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox_fixed no_local facts tables cannot_change max_unbox_size
    false.

(* The rebuilt unit: [x10] is its field, the variable 61; the call is given
   poison and returns to a wrapper, the parameter of 20 having no use. *)
Definition rebuilt : expr :=
  Let_static
    [Bind_code 50
       (Code [40] 41 3 4
          (make_code_metadata [Singleton Value] [Singleton Value])
          (return_to 3 [int 0]))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Simple (int 5))
          (Let 11 Value (Simple (Var 61))
             (Let_cont_non_recursive 20
                (Handler [] (make_handler_metadata [])
                   (Let 31 Value (make_block [Var 60; Var 11])
                      (return_to 1 [Var 31])))
                (Let_cont_non_recursive 62
                   (Handler [62] (make_handler_metadata [Value])
                      (return_to 20 []))
                   (Apply (call_50 [Const (Poison Value)] 62))))))).

End PoisonArgExample.

Definition poison_arg_example_statement : Prop :=
  PoisonArgExample.to_unbox = [Node_variable 10%positive]
  ∧ mem_node (Node_code_id 50%positive) PoisonArgExample.cannot_change = true
  ∧ arguments_used_by_known_arity_call PoisonArgExample.facts
      (solution_of_tables PoisonArgExample.tables)
      (Node_variable 60%positive) [Var 10%positive]
    = [false]
  ∧ calls_unboxing PoisonArgExample.facts PoisonArgExample.tables
      PoisonArgExample.cannot_change PoisonArgExample.code_deps
      PoisonArgExample.dominated PoisonArgExample.to_unbox
      PoisonArgExample.unit_expr
    = true
  ∧ check_unboxed_reads PoisonArgExample.facts PoisonArgExample.tables
      PoisonArgExample.dominated PoisonArgExample.to_unbox
    = true
  ∧ rebuild_unit_unbox_fixed PoisonArgExample.facts PoisonArgExample.tables
      PoisonArgExample.cannot_change PoisonArgExample.code_deps
      PoisonArgExample.dominated PoisonArgExample.to_unbox true
      PoisonArgExample.unit_expr
    = PoisonArgExample.rebuilt
  ∧ program_refines_observably PoisonArgExample.unit_expr
      PoisonArgExample.rebuilt.

(* The unit of FlambdaReaperUnboxingSpec.escaping_argument_unit, the fault
   of the compiler that gives an unboxed argument to a parameter with a
   use of code that keeps its calling convention: with the tables that the
   rules of the fix give, nothing is unboxed and the unit is of the
   fragment, its call of its parameter [x40] a call whose code is not
   known; with those that the rules of the branch give, [x10] is unboxed
   and given to a parameter that the mask of the call keeps, and the unit
   is not of the fragment. The unit rebuilt in the mode of the fix refines
   the unit. *)
Definition escaping_argument_calls_statement : Prop :=
  let F := FlambdaReaperUnboxingSpec.escaping_argument_facts in
  let T := FlambdaReaperUnboxingSpec.escaping_argument_tables in
  let cc := FlambdaReaperUnboxingSpec.escaping_argument_cannot_change in
  let cd := FlambdaReaperUnboxingSpec.escaping_argument_code_deps in
  let dom := FlambdaReaperUnboxingSpec.escaping_argument_dominated in
  let e := FlambdaReaperUnboxingSpec.escaping_argument_unit in
  let to_unbox := FlambdaReaperUnboxingSpec.escaping_argument_to_unbox in
  calls_unboxing F T cc cd dom (to_unbox true) e = true
  ∧ check_unboxed_reads F T dom (to_unbox true) = true
  ∧ arguments_used_by_known_arity_call F (solution_of_tables T)
      (Node_variable 60%positive) [Var 10%positive]
    = [true]
  ∧ calls_unboxing F T cc cd dom (to_unbox false) e = false
  ∧ program_refines_observably e
      (rebuild_unit_unbox_fixed F T cc cd dom (to_unbox true) true e).
