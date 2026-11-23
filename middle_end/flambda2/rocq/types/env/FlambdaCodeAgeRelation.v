(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*           Mark Shinwell and Leo White, Jane Street Europe              *)
(*                                                                        *)
(*   Copyright 2019 Jane Street Group LLC                                 *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* This file transcribes, definition by definition and with the compiler's
   names, middle_end/flambda2/types/env/code_age_relation.ml of this
   repository: the type of a code age relation, [empty], [add],
   [get_older_version_of], [has_older_version_in], [meet_set], [meet],
   [union] and [all_code_ids_for_export]. The comment below says what is
   left out and what differs.

   The header above is that of
   middle_end/flambda2/types/env/code_age_relation.ml. *)

(* The code age relation: of which code a piece of code is a newer
   version.

   This file is to be reviewed, line by line against
   middle_end/flambda2/types/env/code_age_relation.ml, as FlambdaModel.v
   is against flambda.mli. It holds definitions only. Nothing here is
   extracted, and no function of the tree calls these but those of
   FlambdaCodeAgeRelationSpec.v and of FlambdaCodeAgeCallsSpec.v: the
   terms of the model do not say of which code their code is a newer
   version (FlambdaModel.code_metadata has no such field), so no relation
   is read off a term. FlambdaCodeAgeRelationSpec.v has the order that a
   relation stands for and what these functions answer in terms of it,
   and FlambdaCodeAgeCallsSpec.v a machine that is given a relation.

   The module has the compiler's name, so that
   [Code_age_relation.meet_set] is that function.

   What the model has, and how it differs from the compiler:
   - [t] is a map from code ids to code ids, [Code_id.t Code_id.Map.t]:
     if it maps [id1] to [id2], then [id1] is a newer version of [id2].
     Here it is a map of FlambdaMapSpec.v, a list of pairs of which the
     first one for a code id counts. [add] puts a pair in front, which is
     [Code_id.Map.add] as [lookup] sees it: the pair that the code id had
     is shadowed. [add] checks nothing, as in the compiler, whose comment
     asks for a check of well-formedness there
     (FlambdaCodeAgeRelationSpec.well_formed and the statements on [add]).
   - A set of code ids, [Code_id.Set.t], is a list of code ids in which
     order and repetition mean nothing (FlambdaMapSpec.v). [Set.mem] is
     [among]; [Set.equal] is [all_among] both ways; [Set.union] is the
     two lists one after the other; [Set.filter] is [filter_ids] below;
     [Set.is_empty] is the test for the empty list.
   - [has_older_version_in] is a recursive function of the compiler that
     has no measure: it follows the relation from a code id to its older
     version until it finds a candidate or a code id that has no older
     version, and on a relation with a cycle that reaches no candidate it
     does not return. Here it takes a number of steps, one for each code
     id that it looks at, and answers [None] when they run out. [meet_set]
     and [meet] take the number and give it to each of their calls of
     [has_older_version_in]; an answer [None] of one of those calls is an
     answer [None] of theirs. So these are bounded models of the three
     functions, and say nothing of an input on which the compiler's
     function does not return. FlambdaCodeAgeRelationSpec.v says how many
     steps are enough.
   - Compilation units. A code id of the model is a number and has no
     compilation unit; the typing environment of the model has one unit
     and reads no .cmx file (FlambdaTypingEnv.v). The three functions
     ask for the unit of a code id, compare it with the current unit and
     ask the resolver for the relation of another unit. Here those are
     parameters, the variables of the section [Units]: the type of
     compilation units, [Code_id.get_compilation_unit],
     [Current_unit.get_cu_exn ()] and [Compilation_unit.equal]. The
     resolver is an argument, as in the compiler; it is a function, and
     the fatal error of the compiler on a resolver that raises is left
     out. After the resolver has given the relation of another unit, the
     compiler goes on in that relation and not in the first one; so
     does this.
   - [should_keep_id] is a local function of [meet_set] in the compiler;
     here it is a definition before it. It keeps a code id of which
     [has_older_version_in] answers [Unknown].
   - [union] is [Code_id.Map.disjoint_union ~eq:Code_id.equal], which
     raises where the two maps give one code id two different older
     versions; here the answer is then [None].
   - [all_code_ids_for_export] answers a list in which a code id may be
     repeated.
   - [print], [apply_renaming] and [clean_for_export] are left out: the
     model prints nothing, renames no code id (FlambdaNominalSpec.v), and
     has no [Name_occurrences.t]. *)

From Stdlib Require Import PArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar.
Import ListNotations.

Module Code_age_relation.

  (* [Code_age_relation.t]. If a value of type [t] maps [id1] to [id2], it
     means that [id1] is a newer version of [id2]. *)
  Definition t : Type := list (code_id * code_id).

  Definition empty : t := [].

  Definition get_older_version_of (rel : t) (id : code_id)
      : option code_id :=
    lookup id rel.

  Definition add (rel : t) (newer older : code_id) : t :=
    (newer, older) :: rel.

  Section Units.

  (* What the compiler knows of compilation units: their type, the unit
     of a code id, the unit that is being compiled, and their equality. *)
  Variable compilation_unit : Type.
  Variable get_compilation_unit : code_id -> compilation_unit.
  Variable current_unit : compilation_unit.
  Variable compilation_unit_equal :
    compilation_unit -> compilation_unit -> bool.

  (* [has_older_version_in], in at most [steps] steps: whether [id] or an
     older version of it is one of [candidates]. [None] when the steps
     run out. *)
  Fixpoint has_older_version_in (steps : nat) (rel : t)
      (resolver : compilation_unit -> option t) (id : code_id)
      (candidates : list code_id) {struct steps}
      : option (Or_unknown.t bool) :=
    match steps with
    | O => None
    | S steps =>
      if among id candidates
      then Some (Or_unknown.Known true)
      else
        match lookup id rel with
        | None =>
          let comp_unit := get_compilation_unit id in
          if compilation_unit_equal comp_unit current_unit
          then Some (Or_unknown.Known false)
          else
            match resolver comp_unit with
            | None => Some Or_unknown.Unknown
            | Some rel =>
              (* Inlining the base case, so that we do not recursively
                 loop in case of a code_id that is not bound in the map *)
              match lookup id rel with
              | None => Some (Or_unknown.Known false)
              | Some older =>
                has_older_version_in steps rel resolver older candidates
              end
            end
        | Some older =>
          has_older_version_in steps rel resolver older candidates
        end
    end.

  (* The local function [should_keep_id] of [meet_set]: whether [id] or
     an older version of it is one of [other_ids], and [true] where that
     is not known. *)
  Definition should_keep_id (steps : nat) (rel : t)
      (resolver : compilation_unit -> option t) (other_ids : list code_id)
      (id : code_id) : option bool :=
    match has_older_version_in steps rel resolver id other_ids with
    | Some (Or_unknown.Known has_older_version_in_other_ids) =>
      Some has_older_version_in_other_ids
    | Some Or_unknown.Unknown => Some true
    | None => None
    end.

  (* [Code_id.Set.filter], by a test that may have no answer: then there
     is no answer. *)
  Fixpoint filter_ids (keep : code_id -> option bool) (ids : list code_id)
      : option (list code_id) :=
    match ids with
    | [] => Some []
    | id :: ids =>
      let* kept := keep id in
      let* ids := filter_ids keep ids in
      Some (if kept then id :: ids else ids)
    end.

  (* [meet_set]. We are trying to keep a reasonably precise set of the
     code ids that can actually be called, in order to avoid de-optimizing
     to a single direct call.

     The typical case is that we could meet { A, A1, A2 } with { A3, A4 }
     where A1, A2, A3, A4 are newer versions of A, in which case we want
     the output to be { A3, A4 } (from the second set, only newer versions
     of A3 and A4 are possible, which excludes newer versions of A1 and
     A2).

     Returning { A } would be correct, but imprecise. *)
  Definition meet_set (steps : nat) (rel : t)
      (resolver : compilation_unit -> option t) (ids1 ids2 : list code_id)
      : option (Or_bottom.t (list code_id)) :=
    if all_among ids1 ids2 && all_among ids2 ids1
    then Some (Or_bottom.Ok ids1)
    else
      let* kept2 := filter_ids (should_keep_id steps rel resolver ids1) ids2 in
      let* kept1 := filter_ids (should_keep_id steps rel resolver ids2) ids1 in
      match kept2 ++ kept1 with
      | [] => Some Or_bottom.Bottom
      | ids => Some (Or_bottom.Ok ids)
      end.

  (* [meet]. Whichever of [id1] and [id2] is newer (or the same as the
     other one), in the case where they are comparable; otherwise
     bottom. *)
  Definition meet (steps : nat) (rel : t)
      (resolver : compilation_unit -> option t) (id1 id2 : code_id)
      : option (or_unknown_or_bottom code_id) :=
    if Pos.eqb id1 id2
    then Some (Ok id1)
    else
      let* older1 := has_older_version_in steps rel resolver id2 [id1] in
      let* older2 := has_older_version_in steps rel resolver id1 [id2] in
      Some
        match older1, older2 with
        | Or_unknown.Known true, _ =>
          (* id1 is an older version of id2 -> id2 is more precise *)
          Ok id2
        | _, Or_unknown.Known true =>
          (* id2 is an older version of id1 -> id1 is more precise *)
          Ok id1
        | Or_unknown.Known false, Or_unknown.Known false =>
          (* id1 and id2 are definitely not related *)
          Bottom
        | Or_unknown.Unknown, _ | _, Or_unknown.Unknown =>
          (* can't determine whether id1 and id2 are related *)
          Unknown
        end.

  End Units.

  (* [union]: [None] where the compiler's [disjoint_union] raises. *)
  Definition union (rel1 rel2 : t) : option t :=
    if
      forallb
        (fun id =>
           match lookup id rel1, lookup id rel2 with
           | Some older1, Some older2 => Pos.eqb older1 older2
           | _, _ => true
           end)
        (map fst rel1)
    then Some (rel1 ++ rel2)
    else None.

  Definition all_code_ids_for_export (rel : t) : list code_id :=
    flat_map (fun binding => [fst binding; snd binding]) (compact rel).

End Code_age_relation.
